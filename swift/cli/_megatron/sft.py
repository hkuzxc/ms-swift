# Copyright (c) ModelScope Contributors. All rights reserved.
import os

if __name__ == '__main__':
    os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
    import pdb; pdb.set_trace()  # 断点1: 程序入口
    from swift.megatron import megatron_sft_main
    megatron_sft_main()
