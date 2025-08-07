BackendCompilerFailed: backend='inductor' raised:
LoweringException: ImportError: cannot import name 'ir' from 'triton._C.libtriton' (/home/sysop/.local/lib/python3.10/site-packages/triton/_C/libtriton.so)
  target: aten.mm.default
  args[0]: TensorBox(StorageBox(
    ComputedBuffer(name='buf5', layout=FixedLayout('cuda', torch.float16, size=[8448, 384], stride=[384, 1]), data=Pointwise(
      'cuda',
      torch.float16,
      def inner_fn(index):
          i0, i1 = index
          tmp0 = ops.load(primals_97, i1 + 384 * ModularIndexing(i0, 256, 33) + 12672 * ModularIndexing(i0, 1, 256))
          tmp1 = ops.load(buf0, 33 * ModularIndexing(i0, 1, 256) + ModularIndexing(i0, 256, 33))
          tmp2 = tmp0 - tmp1
          tmp3 = ops.load(buf3, 33 * ModularIndexing(i0, 1, 256) + ModularIndexing(i0, 256, 33))
          tmp4 = tmp2 * tmp3
          tmp5 = ops.load(primals_9, i1)
          tmp6 = tmp4 * tmp5
          tmp7 = ops.load(primals_10, i1)
          tmp8 = tmp6 + tmp7
          tmp9 = ops.to_dtype(tmp8, torch.float16, src_dtype=torch.float32)
          return tmp9
      ,
      ranges=[8448, 384],
      origin_node=view,
      origins={clone, convert_element_type_2, view}
    ))
  ))
  args[1]: TensorBox(StorageBox(
    ComputedBuffer(name='buf4', layout=FixedLayout('cuda', torch.float16, size=[384, 1152], stride=[1, 384]), data=Pointwise(
      'cuda',
      torch.float16,
      def inner_fn(index):
          i0, i1 = index
          tmp0 = ops.load(primals_1, i0 + 384 * i1)
          tmp1 = ops.to_dtype(tmp0, torch.float16, src_dtype=torch.float32)
          return tmp1
      ,
      ranges=[384, 1152],
      origin_node=permute_1,
      origins={permute_1, convert_element_type_1}
    ))
  ))

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

Click to add a cell.
