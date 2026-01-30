---
name: conda-skills
description: conda code ruler
---

# gen ruler

在编写代码进行执行测试时必须使用本地conda命令去先激活虚拟环境，python的版本必须是3.10.x，因为python3.10这个版本的兼容性最好，python版本过高会导致深度学习相关的库无法使用，因为当前操作系统的python是由conda统一管理的，命令为conda activate TensorFlow，此环境专门用于TensorFlow深度学习，所有python第三方库都统一通过pip命令进行依赖管理，比如安装第三方库时也是需要先激活conda虚拟环境然后再去执行pip命令安装依赖