from setuptools import setup, find_packages

setup(
    name="lidar_det",
    version="1.0",
    author="Jinzheng Guang",
    author_email="guangjinzheng@mail.nankai.edu.cn",
    packages=find_packages(
        include=["lidar_det", "lidar_det.*", "lidar_det.*.*"]
    ),
    license="LICENSE.txt",
    description="DCCLA: Dense Cross Connections with Linear Attention for LiDAR-based 3D Pedestrian Detection.",
)
