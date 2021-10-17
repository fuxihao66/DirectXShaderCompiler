auto patch_intrinsic = [&](std::string name, Intrinsic::ID intrin,
                                 bool ret = true,
                                 std::vector<llvm::Type *> types = {},
                                 std::vector<llvm::Value *> extra_args = {}) {
        auto func = runtime_module->getFunction(name);
        TI_ERROR_UNLESS(func, "Function {} not found", name);
        func->deleteBody();
        auto bb = llvm::BasicBlock::Create(*ctx, "entry", func);
        IRBuilder<> builder(*ctx);
        builder.SetInsertPoint(bb);
        std::vector<llvm::Value *> args;
        for (auto &arg : func->args())
          args.push_back(&arg);
        args.insert(args.end(), extra_args.begin(), extra_args.end());
        if (ret) {
          builder.CreateRet(builder.CreateIntrinsic(intrin, types, args));
        } else {
          builder.CreateIntrinsic(intrin, types, args);
          builder.CreateRetVoid();
        }
        TaichiLLVMContext::mark_inline(func);
      };