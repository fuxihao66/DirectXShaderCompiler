opengl
1. 传统
    *   
        ```
        vertex = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex, 1, &vShaderCode, NULL);
        glCompileShader(vertex);
        <!-- checkCompileErrors(vertex, "VERTEX"); -->
        // same for fragment
        ID = glCreateProgram();
        glAttachShader(ID, vertex);
        glAttachShader(ID, fragment);
        glLinkProgram(ID);
        <!-- checkCompileErrors(ID, "PROGRAM"); -->
        glDeleteShader(vertex);
        glDeleteShader(fragment);
        glUseProgram(ID); 
        ```
2. spirv
    如何加载spirv

        ```
        vertex = glCreateShader(GL_VERTEX_SHADER);
        glShaderBinary(1, &vertex, GL_SHADER_BINARY_FORMAT_SPIR_V_ARB, vs_buf, vs_buf_len);
        glSpecializeShader(vertex, "main", 0, 0, 0);
        // same
        ```
vulkan
* spirv

dx
* dxbc
* dxir

如何通过cmake/vs来实现shader编译