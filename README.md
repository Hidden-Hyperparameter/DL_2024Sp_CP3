# My Notes
see ![here](./docs/note.md)

# Original README file
同学们好，以下为Coding Project 3的内容。在CP3中，你将实现四种课上讲过的生成模型。以下为关于本次Coding Project的几点要求：

1. 请按照下发的`<method>.ipynb`中的指示，填写缺失的代码。

2. 请勿修改没有标明可以修改的代码。

3. 请保证提交的`<method>.ipynb`文件是可以完整运行的。

4. 请将提交的模型保存为`<method>/<method>_best.pth`

5. 本次Coding Project提交文件大小限制为200MB。

6. 最终提交的zip文件按照下列目录结构排布，注意不需要上传MnistInceptionV3.pth文件。



```

<StudentID>_<StudentName>.zip

└──CodingProject3

    ├── report.pdf

    ├── flow.ipynb

    ├── ebm.ipynb

    ├── vae.ipynb

    ├── gan.ipynb

    ├── flow

    │   └── flow_best.pth

    ├── ebm

    │   └── ebm_best.pth

    ├── gan

    │   ├── gan_best.pth

    │   └── generated

    │       └── 0

    │           ├── 0_000.png

    │           ├── ...

    │           ├── 0_099.png

    │       └── 1

    │           ├── 1_000.png

    │           ├── ...

    │           ├── 1_099.png

    │       └── ...

    │       └── 9

    │           ├── 9_000.png

    │           ├── ...

    │           ├── 9_099.png

    ├── vae

    │   ├── vae_best.pth

    │   └── generated

    │       └── 0

    │           ├── 0_000.png

    │           ├── ...

    │           ├── 0_099.png

    │       └── 1

    │           ├── 1_000.png

    │           ├── ...

    │           ├── 1_099.png

    │       └── ...

    │       └── 9

    │           ├── 9_000.png

    │           ├── ...

    │           ├── 9_099.png

    ├── ...

```