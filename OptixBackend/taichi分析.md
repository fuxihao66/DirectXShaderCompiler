program.cpp

```c++

FunctionType Program::compile(Kernel &kernel) {
  auto start_t = Time::get_time();
  TI_AUTO_PROF;
  FunctionType ret = nullptr;
  if (arch_is_cpu(kernel.arch) || kernel.arch == Arch::cuda ||
      kernel.arch == Arch::metal) {
    kernel.lower();
    ret = compile_to_backend_executable(kernel, /*offloaded=*/nullptr);
  } else if (kernel.arch == Arch::opengl) {
    opengl::OpenglCodeGen codegen(kernel.name, &opengl_struct_compiled_.value(),
                                  opengl_kernel_launcher_.get());
    ret = codegen.compile(*this, kernel);
#ifdef TI_WITH_CC
  } else if (kernel.arch == Arch::cc) {
    ret = cccp::compile_kernel(&kernel);
#endif
  } else {
    TI_NOT_IMPLEMENTED;
  }
  TI_ASSERT(ret);
  total_compilation_time += Time::get_time() - start_t;
  return ret;
}
```



调用了kernel.lower()  编译出kernel的ir

然后调用compile_to_backend_executable创建codegen  把kernel的ir传递给codegen

```c++
FunctionType Program::compile_to_backend_executable(Kernel &kernel,
                                                    OffloadedStmt *offloaded) {
  if (arch_is_cpu(kernel.arch) || kernel.arch == Arch::cuda) {
    auto codegen = KernelCodeGen::create(kernel.arch, &kernel, offloaded);
    return codegen->compile();
  } else if (kernel.arch == Arch::metal) {
    return metal::compile_to_metal_executable(&kernel, metal_kernel_mgr_.get(),
                                              &metal_compiled_structs_.value(),
                                              offloaded);
  }
  TI_NOT_IMPLEMENTED;
  return nullptr;
}
```



在这里创建codegen，然后调用codegen的compile

```c++
FunctionType KernelCodeGen::compile() {
  TI_AUTO_PROF;
  return codegen();
}
```



codegen的compile调用codegen的虚函数codegen()

以cudacodegen为例，

```c++
FunctionType CodeGenCUDA::codegen() {
  TI_AUTO_PROF
  return CodeGenLLVMCUDA(kernel, ir).gen();
}
```



这里的gen调用的是codegenllvm的gen



```c++
FunctionType CodeGenLLVM::gen() {
  emit_to_module();
  return compile_module_to_executable();
}
```



这里调用了虚函数compile_module_to_executable

```c++
FunctionType compile_module_to_executable() override {
#ifdef TI_WITH_CUDA
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
#else
    TI_ERROR("No CUDA");
    return nullptr;
#endif  // TI_WITH_CUDA
  }
```



因此只需看这最后这个函数是如何把kernel传过来的ir生成对应平台的代码并编译的