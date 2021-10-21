# 路线

### 方向

1. 学习LLVM IR 能用cpp emit出需要的nvvm ir并且成功编译成ptx
2. 从AST出发  仿照SpirvEmitter写codegen（难点：很多debug相关的内容 影响阅读）

### 问题

1. optix和hlsl的声明 等区别（比如加速结构   texture等）

   1. 变量全部wrap成一个param（包括加速结构等）

   2. payload     不同于DXR的TraceRay(T rayPayload)的payload可以使用模板；optix的payload通过不同参数的函数重载实现，最多八个参数

      ```c++
      static __forceinline__ __device__ void optixTrace( OptixTraversableHandle handle,
      float3                 rayOrigin,
      float3                 rayDirection,
      float                  tmin,
      float                  tmax,
      float                  rayTime,
      OptixVisibilityMask    visibilityMask,
      unsigned int           rayFlags,
      unsigned int           SBToffset,
      unsigned int           SBTstride,
      unsigned int           missSBTIndex,
      unsigned int&          p0,
      unsigned int&          p1,
      unsigned int&          p2,
      unsigned int&          p3,
      unsigned int&          p4,
      unsigned int&          p5,
      unsigned int&          p6,
      unsigned int&          p7 );
      ```

      并且在hit和miss中使用

      ```
      optixSetPayload_0
      ```

      写入payload

   3. 既可以使用texture  也可以使用指针（比如使用float4*）   CUtexObject（driver api）或者**cudaTextureObject_t**（runtime api）

   4. texture的采样方式在创建texture的时候就确定了（这样我们shader编译的时候完全不考虑这一点，而是在host端提供类似sampler的接口）

   5. shader binding table 原本需要local RS，optix中使用optixGetSbtDataPointer（一个解决方法是把hlsl中带有local的变量wrap起来，作为sbt data）

      ```c++
      const whitted::HitGroupData* hit_group_data = reinterpret_cast<whitted::HitGroupData*>( optixGetSbtDataPointer() );
          const LocalGeometry          geom           = getLocalGeometry( hit_group_data->geometry_data );
      ```

2. 对比vulkan如何处理上述  https://www.willusher.io/graphics/2019/11/20/the-sbt-three-ways

   sbt使用shaderRecordEXT来声明

   ```
   layout(shaderRecordEXT) buffer SBTData {
       uint32_t material_id;
   };
   ```

   

3. nvvm intrinsic替换

4. optix inline asm替换

   问题

   * 因为hlsl和optix接口的不同，不能直接修改translateIntrinsic，因为它是利用RAUW来替换graph中的操作为对应intrinsic，但是function定义和optix的intrinsic定义不同，不能直接使用RAUW来替换成optix inline asm





# LLVM

lowering: 从Inst到DAG SDNode

replaceAllUsesWith：修改IR，把某种Value全部替换

Value：LLVM IR中所有都是Value，包括Inst Const等



整个流程：

1. parser构建出AST
2. AST到IR
3. IR到DAG（lowering）
4. Instruction Selection到target platform

--------------------------------------------------------------------------------------------------------------------------------------------------------------------



**dxil intrinsic：先创建了callinst，然后替换掉原本的ir中的inst**

创建callinst是如何使用dx.op intrinsic的？？？



问题

1. taichi中处理nvvm intrinsic   是通过CreateIntrinsic，根据intrinsic id自动从td中匹配对应的intrinsic function来创建  （llvmcontext.cpp）
2. 对于dxil？





资料：

1. http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15745-s14/public/lectures/
2. LLVM Tutorial

概念

* pass
* builder
* module
* tableGen，intrinsic
* lowering



https://www.cnblogs.com/Five100Miles/p/12822190.html

![img](https://img2020.cnblogs.com/blog/1335902/202005/1335902-20200510211753210-159436990.jpg)

code emission  https://www.cnblogs.com/Five100Miles/p/12903057.html

**ir instruction** https://www.cnblogs.com/Five100Miles/p/14100555.html

### LLVM Program Structure

* Module 
  * **管理了ir的内存**
  * 包含了list of Functions和GlobalVariables
* Function
  * 包含list of BasicBlocks和Arguments
* BasicBlock
  * list of instructions
* Intruction
  * opcode+vector of operands

Module Function和BasicBlock都由双向链表构成，通过iterator遍历



### LLVM Pass Manager

compiler由一系列passes构成，每个pass是一个analysis或transformation

四种pass

* ModulePass  
* CallGraphSCCPass
* FunctionPass  每次处理一个Function
* BasicBlockPass   每次处理一个basic block

### LLVM IR

* Interpreting An Instruction
  * LoadInst, StoreInst, AllocaInst, CallInst, BranchInst
* Every BasicBlock must end with a terminator instruction
  * ret, br, switch, indirectbr, invoke, resume, and unreachable
* Binary Instructions
* Memory Instructions

### Writing Passes

* Analysis Passes vs. Optimization Passes
  * provide information vs modify the program
* 

### Register Allocation

IR阶段假设有无限的register，但是实际的机器上肯定是有限的

**需要多少register等价于n coloring问题**



# LLVM笔记

module

builder



如何添加一个pass  **https://llvm.org/docs/WritingAnLLVMPass.html**

* ```c++
  runOnModule
  ```



如何添加intrinsic https://zhuanlan.zhihu.com/p/53659330 https://llvm.org/docs/ExtendingLLVM.html#intrinsic-function

* ```text
  Builder.CreateIntrinsic(Intrinsic);
  ```



内联asm 

```c++
llvm::InlineAsm *IA =
    llvm::InlineAsm::get(FTy, AsmString, Constraints, HasSideEffect,
                         /* IsAlignStack */ false, AsmDialect);
  llvm::CallInst *Result = Builder.CreateCall(IA, Args);
  Result->addAttribute(llvm::AttributeSet::FunctionIndex,
                       llvm::Attribute::NoUnwind);
```

参考链接：

https://llvm.org/docs/LangRef.html#input-constraints

https://llvm.org/docs/LangRef.html#inline-assembler-expressions

https://stackoverflow.com/questions/28787799/insert-inline-assembly-expressions-using-llvm-pass





DXIL编译 调用栈

- gLowerTable
- TranslateBuiltinIntrinsic 
- TranslateHLBuiltinOperation
- TranslateBuiltinOperations
- GenerateDxilOperations
- runOnModule (DxilGenerationPass)
- PassManagerImpl::run   //**PerModulePasses->run(*TheModule);**
- EmitAssemblyHelper::EmitAssembly
- EmitBackendOutput
- HandleTranslationUnit
- 



想法：

1. 如果dxc是从ast用llvm的接口来创建dxil的，那应该可以做一定修改来实现创建nvvm？
2. 还是说用taichi里面的方法，直接用compile_module_to_ptx，把llvm::module编译成ptx？



# question

intrinsic    inline asm

把所有过程穿起来

cfg     结合ast到ir的codegen理解

ast到ir和ir到target都叫codegen（ast本身就是一种ir）

* DXC中是从ast直接到spirv   但是dxil-spirv是从dxil转化为spirv
* taichi中nvvm后端是从ast出发的

# Engineering a compiler

basic block和CFG

* 一个Basic Block是指一段串行执行的指令流，除了最后一句之外不会有跳转指令
* cfg描述了basic block之间的关系



ast和DAG区别：

* ast中有重复变量，但是DAG把重复的去除了（所以是graph而不是tree）
* 参考engineering a compiler 2nd pg229

control-flow graph与ast不同

* ast是程序的所有语法信息
* 但是CFG是block和block之间的关系



后端的三个任务（这三个任务合起来叫做code generation）

* 从IR到machine operation
  * instruction selection
* operation的执行顺序
  * instruction scheduling
* which values should reside in registers and which values should reside in memory
  * register alloc





# taichi codegen

```
FunctionType CodeGenLLVM::gen() {
  emit_to_module();
  return compile_module_to_executable();
}
```

这里的emit_to_module中调用了ir->accept

也就调用了codegen的visit函数



compile_module_to_executable中调用compile_module_to_ptx把llvm module编译成ptx

调用这个函数的时候需要一个llvm module，这个是从ast得到的



```c++
else if (op == UnaryOpType::sqrt) {
      if (input_taichi_type->is_primitive(PrimitiveTypeID::f32)) {
        llvm_val[stmt] =
            builder->CreateCall(get_runtime_function("__nv_sqrtf"), input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f64)) {
        llvm_val[stmt] =
            builder->CreateCall(get_runtime_function("__nv_sqrt"), input);
      } else {
        TI_NOT_IMPLEMENTED
      }
```







# DXIL_SPIRV

调用builder_lut中绑定的emit函数（eg: emit_trace_ray_instruction）

emit_dxil_instruction

emit_instruction

convert_function

convert_entry_point

dxil_spv_converter_run

他通过自己实现的一个dxil parser





bool SPIRVModule::Impl::finalize_spirv(Vector<uint32_t> &spirv)

把builder中的instruction dump到spirv binary



**对应DXC的finalizeInstruction**

# LLVM

## LLVM IR

虚拟指令集

* Static Single Assignment (SSA) form. Note that there is no value that is reassigned



## Backend

### Instruction Selection

从IR到DAG

SelectionDAG class

* 每个节点 SDNode对应了一个instruction或者operand（这里的指令已经是硬件相关的？？）
* 每个block需要构建一个DAG



reference

* llvm/include/ llvm/CodeGen/**SelectionDAG.h**

#### Lowering

#### DAG combine and legalization

#### DAG-to-DAG instruction selection



### Scheduler

 ScheduleDAGSDNodes（lib/CodeGen/ SelectionDAG/ScheduleDAGSDNodes.cpp）

分配register

* transform an endless number of virtual registers into physical (limited) ones
* deconstruct the SSA form of the IR

scheduling之后，InstrEmitter pass会把SDNode转化为MachineInstr



**MachineInstr is a three-address representation of the program**

* Each MI holds an opcode number, which is a number that has a meaning only for a specific backend, and a list of operands.



### Code Emission

 从MachineInstr到binary

* AsmPrinter  每次拿出一个MI到EmitInstruction
* EmitInstruction把一个MI转化为machine code instance





## 实际实现backend

https://llvm.org/docs/WritingAnLLVMBackend.html



## DXC源码

执行逻辑：

* dxc.cpp  DxcContext::Compile   CompileWithDebug

* dxcompilerobj.cpp Compile

  * 创建clang::EmitSpirvAction action;
  * action.BeginSourceFile(compiler, file);
  * action.Execute();
  * action.EndSourceFile();

* 在BeginSourceFile中：FrontendAction.cpp

  * std::unique_ptr<ASTConsumer> Consumer = CreateWrappedASTConsumer(CI, InputFile);

  * 这个过程会调用EmitSpirvAction.cpp中的CreateASTConsumer
  * 并返回一个MultiplexConsumer

* 在Execute中
  * FrontendAction::Execute()
  
  * ASTFrontendAction::ExecuteAction
  
  * ParseAST (ParseAST.cpp)
  
  * MultiplexConsumer::HandleTranslationUnit
  
  * SpirvEmitter::HandleTranslationUnit
  
  * SpirvEmitter::doDecl
    * SpirvEmitter::doFunctionDecl
    * SpirvEmitter::doStmt
    * SpirvEmitter::doExpr
    * SpirvEmitter::processIntrinsicCallExpr
    * SpirvEmitter::processTraceRay
    * createRayTracingOpsNV
    * SpirvRayTracingOpNV
    
  * SpirvBuild::takeModule
    
    * SpirvModule::invokeVisitor
    
  * 调用visit 写入binary
  
  * takeBinary把所有binary写入最终buffer
  
  * theCompilerInstance.getOutStream()->write(
  
       reinterpret_cast<const char *>(m.data()), m.size() * 4);



clang/SPIRV： 构建DAG

clang/lib/SPIRV：scheduler和emission



其他地方：

主要是配置

* HLSLOptions.h   
  * bool GenSPIRV;          // OPT_spirv
  * clang::spirv::SpirvCodeGenOptions SpirvOptions;    // 用于配置spirv的编译参数  定义在SPRIVOptions.h中



