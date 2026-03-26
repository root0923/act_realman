hdf5格式的数据存储文件的具体结构如下：

-action: Dataset, shape (400, 14), dtype float32 \
-observations: Group \
    --images: Group \
        ---cam01: Dataset, shape (400, 480, 640, 3), dtype unit8 \
        ---cam02 \
        ---cam... \
    --qpos: Dataset, shape: (400, 14), dtype: float32 \
    --qvel: Dataset, shape: (400, 14), dtype: float32 \