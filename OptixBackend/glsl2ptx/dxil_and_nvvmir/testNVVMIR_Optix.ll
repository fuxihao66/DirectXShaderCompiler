  
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

@a = internal addrspace(3) global i32 0, align 4
@b = internal addrspace(3) global [10 x i32] zeroinitializer, align 4

define void @foo() {
entry:
  store i32 1, i32 addrspace(3)* @a, align 4
  store i32 2, i32 addrspace(3)* getelementptr inbounds ([10 x i32] addrspace(3)* @b, i64 0, i64 5), align 4
  ret void
}

!nvvm.annotations = !{!0}
!0 = metadata !{void ()* @foo, metadata !"kernel", i32 1}



%2 = call i32 asm sideeffect "call (%0), _optix_get_launch_index_x, ();" : "=r"( u0 )