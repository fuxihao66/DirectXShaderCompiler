在JITModule *add_module中编译成了ptx，然后加载





std::string JITSessionCUDA::compile_module_to_ptx(
    std::unique_ptr<llvm::Module> &module) {
  TI_AUTO_PROF
  // Part of this function is borrowed from Halide::CodeGen_PTX_Dev.cpp
  if (llvm::verifyModule(*module, &llvm::errs())) {
    module->print(llvm::errs(), nullptr);
    TI_WARN("Module broken");
  }

  using namespace llvm;

  if (get_current_program().config.print_kernel_llvm_ir) {
    static FileSequenceWriter writer("taichi_kernel_cuda_llvm_ir_{:04d}.ll",
                                     "unoptimized LLVM IR (CUDA)");
    writer.write(module.get());
  }

  for (auto &f : module->globals())
    f.setName(convert(f.getName()));
  for (auto &f : *module)
    f.setName(convert(f.getName()));

  llvm::Triple triple(module->getTargetTriple());

  // Allocate target machine

  std::string err_str;
  const llvm::Target *target =
      TargetRegistry::lookupTarget(triple.str(), err_str);
  TI_ERROR_UNLESS(target, err_str);

  bool fast_math = get_current_program().config.fast_math;

  TargetOptions options;
  options.PrintMachineCode = 0;
  if (fast_math) {
    options.AllowFPOpFusion = FPOpFusion::Fast;
    // See NVPTXISelLowering.cpp
    // Setting UnsafeFPMath true will result in approximations such as
    // sqrt.approx in PTX for both f32 and f64
    options.UnsafeFPMath = 1;
    options.NoInfsFPMath = 1;
    options.NoNaNsFPMath = 1;
  } else {
    options.AllowFPOpFusion = FPOpFusion::Strict;
    options.UnsafeFPMath = 0;
    options.NoInfsFPMath = 0;
    options.NoNaNsFPMath = 0;
  }
  options.HonorSignDependentRoundingFPMathOption = 0;
  options.NoZerosInBSS = 0;
  options.GuaranteedTailCallOpt = 0;
  options.StackAlignmentOverride = 0;

  std::unique_ptr<TargetMachine> target_machine(target->createTargetMachine(
      triple.str(), CUDAContext::get_instance().get_mcpu(), cuda_mattrs(),
      options, llvm::Reloc::PIC_, llvm::CodeModel::Small,
      CodeGenOpt::Aggressive));

  TI_ERROR_UNLESS(target_machine.get(), "Could not allocate target machine!");

  module->setDataLayout(target_machine->createDataLayout());

  // Set up passes
  llvm::SmallString<8> outstr;
  raw_svector_ostream ostream(outstr);
  ostream.SetUnbuffered();

  legacy::FunctionPassManager function_pass_manager(module.get());
  legacy::PassManager module_pass_manager;

  module_pass_manager.add(createTargetTransformInfoWrapperPass(
      target_machine->getTargetIRAnalysis()));
  function_pass_manager.add(createTargetTransformInfoWrapperPass(
      target_machine->getTargetIRAnalysis()));

  // NVidia's libdevice library uses a __nvvm_reflect to choose
  // how to handle denormalized numbers. (The pass replaces calls
  // to __nvvm_reflect with a constant via a map lookup. The inliner
  // pass then resolves these situations to fast code, often a single
  // instruction per decision point.)
  //
  // The default is (more) IEEE like handling. FTZ mode flushes them
  // to zero. (This may only apply to single-precision.)
  //
  // The libdevice documentation covers other options for math accuracy
  // such as replacing division with multiply by the reciprocal and
  // use of fused-multiply-add, but they do not seem to be controlled
  // by this __nvvvm_reflect mechanism and may be flags to earlier compiler
  // passes.
  const auto kFTZDenorms = 1;

  // Insert a module flag for the FTZ handling.
  module->addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz",
                        kFTZDenorms);

  if (kFTZDenorms) {
    for (llvm::Function &fn : *module) {
      fn.addFnAttr("nvptx-f32ftz", "true");
    }
  }

  PassManagerBuilder b;
  b.OptLevel = 3;
  b.Inliner = createFunctionInliningPass(b.OptLevel, 0, false);
  b.LoopVectorize = false;
  b.SLPVectorize = false;

  target_machine->adjustPassManager(b);

  b.populateFunctionPassManager(function_pass_manager);
  b.populateModulePassManager(module_pass_manager);

  // Override default to generate verbose assembly.
  target_machine->Options.MCOptions.AsmVerbose = true;

  // Output string stream

  // Ask the target to add backend passes as necessary.
  bool fail = target_machine->addPassesToEmitFile(
      module_pass_manager, ostream, nullptr, llvm::CGFT_AssemblyFile, true);

  TI_ERROR_IF(fail, "Failed to set up passes to emit PTX source\n");

  {
    TI_PROFILER("llvm_function_pass");
    function_pass_manager.doInitialization();
    for (llvm::Module::iterator i = module->begin(); i != module->end(); i++)
      function_pass_manager.run(*i);

    function_pass_manager.doFinalization();
  }

  {
    TI_PROFILER("llvm_module_pass");
    module_pass_manager.run(*module);
  }

  if (get_current_program().config.print_kernel_llvm_ir_optimized) {
    static FileSequenceWriter writer(
        "taichi_kernel_cuda_llvm_ir_optimized_{:04d}.ll",
        "optimized LLVM IR (CUDA)");
    writer.write(module.get());
  }

  std::string buffer(outstr.begin(), outstr.end());

  // Null-terminate the ptx source
  buffer.push_back(0);
  return buffer;
}






FunctionType compile_module_to_executable() override {
    eliminate_unused_functions();

    auto offloaded_local = offloaded_tasks;
    for (auto &task : offloaded_local) {
      llvm::Function *func = module->getFunction(task.name);
      TI_ASSERT(func);
      tlctx->mark_function_as_cuda_kernel(func, task.block_dim);
    }

    auto jit = kernel->program.llvm_context_device->jit.get();
    auto cuda_module =
        jit->add_module(std::move(module), kernel->program.config.gpu_max_reg);

    return [offloaded_local, cuda_module,
            kernel = this->kernel](Context &context) {
      // copy data to GRAM
      CUDAContext::get_instance().make_current();
      auto args = kernel->args;
      std::vector<void *> host_buffers(args.size(), nullptr);
      std::vector<void *> device_buffers(args.size(), nullptr);
      bool has_buffer = false;

      // We could also use kernel->make_launch_context() to create
      // |ctx_builder|, but that implies the usage of Program's context. For the
      // sake of decoupling, let's not do that and explicitly set the context we
      // want to modify.
      Kernel::LaunchContextBuilder ctx_builder(kernel, &context);
      for (int i = 0; i < (int)args.size(); i++) {
        if (args[i].is_nparray) {
          has_buffer = true;
          // replace host buffer with device buffer
          host_buffers[i] = context.get_arg<void *>(i);
          if (args[i].size > 0) {
            // Note: both numpy and PyTorch support arrays/tensors with zeros
            // in shapes, e.g., shape=(0) or shape=(100, 0, 200). This makes
            // args[i].size = 0.
            CUDADriver::get_instance().malloc(&device_buffers[i], args[i].size);
            CUDADriver::get_instance().memcpy_host_to_device(
                (void *)device_buffers[i], host_buffers[i], args[i].size);
          }
          ctx_builder.set_arg_nparray(i, (uint64)device_buffers[i],
                                      args[i].size);
        }
      }
      if (has_buffer) {
        CUDADriver::get_instance().stream_synchronize(nullptr);
      }

      for (auto task : offloaded_local) {
        TI_TRACE("Launching kernel {}<<<{}, {}>>>", task.name, task.grid_dim,
                 task.block_dim);
        cuda_module->launch(task.name, task.grid_dim, task.block_dim,
                            task.shmem_bytes, {&context});
      }
      // copy data back to host
      if (has_buffer) {
        CUDADriver::get_instance().stream_synchronize(nullptr);
      }
      for (int i = 0; i < (int)args.size(); i++) {
        if (args[i].is_nparray && args[i].size > 0) {
          CUDADriver::get_instance().memcpy_device_to_host(
              host_buffers[i], (void *)device_buffers[i], args[i].size);
          CUDADriver::get_instance().mem_free((void *)device_buffers[i]);
        }
      }
    };

  }