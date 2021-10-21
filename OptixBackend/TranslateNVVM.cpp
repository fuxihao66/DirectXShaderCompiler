



Value *TranslateSample(CallInst *CI, IntrinsicOp IOP, OP::OpCode opcode,
                       HLOperationLowerHelper &helper,  HLObjectOperationLowerHelper *pObjHelper, bool &Translated) {
  hlsl::OP *hlslOP = &helper.hlslOP;
  SampleHelper sampleHelper(CI, opcode, pObjHelper);

  if (sampleHelper.opcode == DXIL::OpCode::NumOpCodes) {
    Translated = false;
    return nullptr;
  }
  Type *Ty = CI->getType();

  Function *F = hlslOP->GetOpFunc(opcode, Ty->getScalarType());

  Constant *opArg = hlslOP->GetU32Const((unsigned)opcode);

  switch (opcode) {
  case OP::OpCode::Sample: {
    Value *sampleArgs[] = {
        opArg, sampleHelper.texHandle, sampleHelper.samplerHandle,
        // Coord.
        sampleHelper.coord[0], sampleHelper.coord[1], sampleHelper.coord[2],
        sampleHelper.coord[3],
        // Offset.
        sampleHelper.offset[0], sampleHelper.offset[1], sampleHelper.offset[2],
        // Clamp.
        sampleHelper.clamp};
    GenerateDxilSample(CI, F, sampleArgs, sampleHelper.status, hlslOP);
  } break;
  case OP::OpCode::SampleLevel: {
    Value *sampleArgs[] = {
        opArg, sampleHelper.texHandle, sampleHelper.samplerHandle,
        // Coord.
        sampleHelper.coord[0], sampleHelper.coord[1], sampleHelper.coord[2],
        sampleHelper.coord[3],
        // Offset.
        sampleHelper.offset[0], sampleHelper.offset[1], sampleHelper.offset[2],
        // LOD.
        sampleHelper.lod};
    GenerateDxilSample(CI, F, sampleArgs, sampleHelper.status, hlslOP);
  } break;
  case OP::OpCode::SampleGrad: {
    Value *sampleArgs[] = {
        opArg, sampleHelper.texHandle, sampleHelper.samplerHandle,
        // Coord.
        sampleHelper.coord[0], sampleHelper.coord[1], sampleHelper.coord[2],
        sampleHelper.coord[3],
        // Offset.
        sampleHelper.offset[0], sampleHelper.offset[1], sampleHelper.offset[2],
        // Ddx.
        sampleHelper.ddx[0], sampleHelper.ddx[1], sampleHelper.ddx[2],
        // Ddy.
        sampleHelper.ddy[0], sampleHelper.ddy[1], sampleHelper.ddy[2],
        // Clamp.
        sampleHelper.clamp};
    GenerateDxilSample(CI, F, sampleArgs, sampleHelper.status, hlslOP);
  } break;
  case OP::OpCode::SampleBias: {
    Value *sampleArgs[] = {
        opArg, sampleHelper.texHandle, sampleHelper.samplerHandle,
        // Coord.
        sampleHelper.coord[0], sampleHelper.coord[1], sampleHelper.coord[2],
        sampleHelper.coord[3],
        // Offset.
        sampleHelper.offset[0], sampleHelper.offset[1], sampleHelper.offset[2],
        // Bias.
        sampleHelper.bias,
        // Clamp.
        sampleHelper.clamp};
    GenerateDxilSample(CI, F, sampleArgs, sampleHelper.status, hlslOP);
  } break;
  case OP::OpCode::SampleCmp: {
    Value *sampleArgs[] = {
        opArg, sampleHelper.texHandle, sampleHelper.samplerHandle,
        // Coord.
        sampleHelper.coord[0], sampleHelper.coord[1], sampleHelper.coord[2],
        sampleHelper.coord[3],
        // Offset.
        sampleHelper.offset[0], sampleHelper.offset[1], sampleHelper.offset[2],
        // CmpVal.
        sampleHelper.compareValue,
        // Clamp.
        sampleHelper.clamp};
    GenerateDxilSample(CI, F, sampleArgs, sampleHelper.status, hlslOP);
  } break;
  case OP::OpCode::SampleCmpLevelZero:
  default: {
    DXASSERT(opcode == OP::OpCode::SampleCmpLevelZero, "invalid sample opcode");
    Value *sampleArgs[] = {
        opArg, sampleHelper.texHandle, sampleHelper.samplerHandle,
        // Coord.
        sampleHelper.coord[0], sampleHelper.coord[1], sampleHelper.coord[2],
        sampleHelper.coord[3],
        // Offset.
        sampleHelper.offset[0], sampleHelper.offset[1], sampleHelper.offset[2],
        // CmpVal.
        sampleHelper.compareValue};
    GenerateDxilSample(CI, F, sampleArgs, sampleHelper.status, hlslOP);
  } break;
  }
  // CI is replaced in GenerateDxilSample.
  return nullptr;
}
