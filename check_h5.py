import h5py

# 打开H5文件
with h5py.File('/home/ysy/data/simple_test/0000.h5', 'r') as f:
    # 查看文件中的所有键（类似于目录结构）
    def print_h5_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"字段: {name} | 形状: {obj.shape} | 类型: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"组: {name}")
    
    f.visititems(print_h5_structure)
    
    # 或者使用递归函数查看完整结构
    def print_all(name, obj):
        print(name)
    f.visititems(print_all)