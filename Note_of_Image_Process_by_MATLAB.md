## 4 Geometric Transformation of Images

### 4.2 Image translation transformation

$$
\begin{pmatrix}
x_{1} \\ y_{1} \\ 1
\end{pmatrix}
=
\begin{pmatrix}
1 & 0 & T_{x} \\
0 & 1 & T_{y} \\
0 & 0 & 1
\end{pmatrix}
\begin{pmatrix}
x_{0} \\ y_{0} \\ 1
\end{pmatrix}
$$

```m
% strel     创建形态学结构元素
A = imread("test.bmp");
se = translate(strel(1), [80, 50]);
% imdilate  形态学膨胀
B = imdilate(A, se);
figure;
subplot(1,2,1), subimage(A);
title("Original Image");
subplot(1,2,2), subimage(B);
title("Tanslated Image");
```

### 4.3 Mirror image

* Mirror horizontal
    $$
    \begin{pmatrix}
    x_{1} \\ y_{1} \\ 1
    \end{pmatrix}
    =
    \begin{pmatrix}
    Width-x_{0} \\ y_{0} \\ 1
    \end{pmatrix}
    $$

* Mirror vertical
    $$
    \begin{pmatrix}
    x_{1} \\ y_{1} \\ 1
    \end{pmatrix}
    =
    \begin{pmatrix}
    x_{0} \\ Height-y_{0} \\ 1
    \end{pmatrix}
    $$

```m
% TFORM     空间变换结构
% method    变换的插值方法
%   1. 'nearest'  最近邻插值
%   2. 'bilinear' 双线性插值
%   3. 'bicubic'  三次插值
B = imtransform(A, TFORM, method);
% 获得 TFORM 结构的方法
T = maketform(transformtype, Matrix);
% Mirror horizontal
A = imread("test.bmp");
[height, width, dim] = size(A);
% 'affine'  表示仿射变换
tform1 = maketform('affine', [-1 0 0; 0 1 0; width 0 1]);
tform2 = maketform('affine', [1 0 0; 0 -1 0; 0 height 1]);
B = imtransform(A, tform1, 'nearest');
C = imtransform(A, tform2, 'nearest');
```

### 4.4 Image transposition

$$
\begin{pmatrix}
x_{1} \\ y_{1} \\ 1
\end{pmatrix}
=
\begin{pmatrix}
0 & 1 & 0 \\
1 & 0 & 0 \\
0 & 0 & 1
\end{pmatrix}
\begin{pmatrix}
x_{0} \\ y_{0} \\ 1
\end{pmatrix}
=
\begin{pmatrix}
y_{0} \\ x_{0} \\ 1
\end{pmatrix}
$$

```m
A = imread("test.bmp");
tform = maketform("affine", [0 1 0; 1 0 0; 0 0 1]);
B = imtransform(A, tform, 'nearest');
```

### 4.5 Resize image

$$
\begin{pmatrix}
x_{1} \\ y_{1} \\ 1
\end{pmatrix}
=
\begin{pmatrix}
S_{x} & 0 & 0 \\
0 & S_{y} & 0 \\
0 & 0 & 1
\end{pmatrix}
\begin{pmatrix}
x_{0} \\ y_{0} \\ 1
\end{pmatrix}
=
\begin{pmatrix}
S_{x}x_{0} \\ S_{y}y_{0} \\ 1
\end{pmatrix}
$$

```m
B = imresize(A, Scale, method);
```

### 4.6 Rotate image

$$
\begin{pmatrix}
x_{1} \\ y_{1} \\ 1
\end{pmatrix}
=
\begin{pmatrix}
\cos\theta & -\sin\theta & 0 \\
\sin\theta & \cos\theta & 0 \\
0 & 0 & 1
\end{pmatrix}
\begin{pmatrix}
x_{0} \\ y_{0} \\ 1
\end{pmatrix}
$$

### 4.7 Interpolation algorithm

* Bilinear interpolation

    ```m
    f(x, 0) = f(0, 0) + x[f(1, 0) - f(0, 0)]
    f(x, 1) = f(0, 1) + x[f(1, 1) - f(0, 1)]
    f(x, y) = f(x, 0) + y[f(x, 1) - f(x, 0)]
    ```

* High order interpolation

    ```m
    % 3 methods of rotating image
    A = imread("test.bmp");
    B = imrotate(A, 30, 'nearest');
    C = imrotate(A, 30, 'bilinear');
    C = imrotate(A, 30, 'bicubic');
    ```

### 4.8 Image matching for adjusting

```m
% Iin   输入图像
% Ibase 基准图像
cpselect(Iin, Ibase);
% 之后可以通过手动选择得到：
%   input_points    输入图像上的点
%   base_points     基准图像上的点
tform = cp2tform(input_points, base_points, 'affine');
```

---

## 5 Image enhancement in space domain

### 5.1 Introduction

Image enhancement includes histogram, gray transformation, image smoothing and image shaping, etc.

### 5.2 Spatial domain filtering

* Process neighborhood
    
    A digital image can be viewed as a 2D-function $f(x,y)$.<br>
    Then the filter operation can be viewed as: (It means relevant)

    $$
    g(x,y)=\sum^{a}_{s=-a}\sum^{b}_{t=-b}w(s,t)f(x+s,y+t)
    $$

* Process boundary
    * 收缩处理范围，即忽略图像周围点的处理
    * 使用常数填充图像，用常数补充一个边界
    * 使用复制像素的方法填充图像

* Convolution method

    $$
    g(x,y)=\sum^{a}_{s=-a}\sum^{b}_{t=-b}w(-s,-t)f(x+s,y+t)
    $$
    Relevant and convolution are both linear algorithm.

* Examples in MATLAB

    ```m
    % 滤波函数
    % f     需要进行滤波操作的图像
    % w     滤波操作使用的模板
    % option    可选项，主要包括：
    %   1. 边界选项：
    %       * 'symmetric'   镜像边缘填充虚拟边界
    %       * 'replicate'   重复最近边缘填充边界
    %       * 'circular'    周期性填充虚拟边界
    %   2. 尺寸选项：
    %       * 'same'        输出图像和输入图像尺寸相同
    %       * 'full'        输出图像尺寸为填充虚拟边界后的尺寸
    %   3. 模式选项：
    %       * 'corr'        相关滤波
    %       * 'conv'        卷积滤波
    g = imfilter(f, w, option1, option2, ...);
    % Example:
    f = imread('test.tif');
    w = [1 1 1; 1 1 1; 1 1 1] / 9;
    g = imfilter(f, w, 'corr', 'replicate');

    % We can define the filter by following code.
    % type  滤波器类型：
    %   * 'average'     平均模板
    %   * 'disk'        圆形邻域模板
    %   * 'gaussian'    高斯模板
    %   * 'laplacian'   拉普拉斯模板
    %   * 'log'         高斯-拉普拉斯模板
    %   * 'prewitt'     Prewitt 水平边缘检测算子
    %   * 'sobel'       Sobel 水平边缘检测算子
    h = fspecial(type, parameters)
    % Examples:
    h = fspecial('average', [3, 3]);
    h = fspecial('disk', 5);
    h = fspecial('gaussian', [3, 3], 0.5);
    h = fspecial('sobel');
    % Vertical operator of sobel is: h'
    ```

### 5.3 Image smoothing

* Average template
    
    主要因为一般情况下，图像具有局部连续的性质。
    $$
    w = \frac{1}{(2k+1)^{2}}
    \begin{pmatrix}
    1 & 1 & \cdots & 1 \\
    1 & 1 & \cdots & 1 \\
    \vdots \\
    1 & 1 & \cdots & 1
    \end{pmatrix}_{(2k+1)\times(2k+1)}
    $$

    ```m
    I = imread('test.bmp');
    h = fspecial('average', 3);
    I3 = imfilter(I, h, 'corr', 'replicate');
    ```

* Gaussian template

    $$
    w = \frac{1}{16}\times
    \begin{pmatrix}
    1 & 2 & 1 \\
    2 & 4 & 2 \\
    1 & 2 & 1
    \end{pmatrix}
    $$

    2D-Gaussian distribution:
    
    $$
    \varphi(x,y)=\frac{1}{2\pi\sigma^{2}}\exp(-\frac{x^2+y^2}{2\sigma^2})
    $$

    Then $(2k+1)\times(2k+1)$ Gaussian template can be defined by:

    $$
    M(i,j)=\frac{1}{2\pi\sigma^{2}}\exp(-\frac{(i-k+1)^2+(j-k+1)^2}{2\sigma^2})
    $$

    ```m
    % Kernal size is 3, sigma is 0.5
    I = imread('test.bmp');
    h3_5 = fspecial('gaussian', 3, 0.5);
    I3_5 = imfilter(I, h3_5);
    ```

* Self-adaption filter

    只在噪声局部区域进行平滑，而在无噪声局部区域不进行平滑，将模糊的影响降到最少。<br>
    局部区域存在噪声的判据：
    1. 局部区域最大值与最小值之差大于某一阈值T
    2. 局部区域方差大于某一阈值T

### 5.4 Median filtering

* Property of Median filter
    * It is not a linear filter.
    * It is a statistical sorting filter.

* Noise model

    ```m
    % type  噪声模型
    %   * 'gaussian'
    %   * 'salt & pepper'
    J = imnoise(I, type, parameters)
    ```

* Median filter

    ```m
    % I = imread('test.bmp');
    J = imnoise(I, 'salt & pepper');
    w = [1 2 1; 2 4 2; 1 2 1] / 16;
    Jm = medfilt2(J, [3, 3]);
    ```

* Self-adaption median filter

    噪声点几乎都是局部邻域的极值，但是边缘点不一定，可以用这种方式限制中值滤波。

### 5.5 Image sharping

* Robert cross gradient

    $$
    w1 =
    \begin{pmatrix}
    -1 & 0 \\
    0 & 1
    \end{pmatrix}
    $$
    $$
    w2 =
    \begin{pmatrix}
    0 & -1 \\
    1 & 0
    \end{pmatrix}
    $$
    $$
    |\nabla f(i,j)|=|f(i+1,j+1)-f(i,j)|+|f(i,j+1)-f(i+1,j)|
    $$
    可以使用 $w1$ 和 $w2$ 为模板对图像进行滤波得到 $G1$ 和 $G2$，之后得到 Robert 交叉梯度为：<br>
    $$G=|G1|+|G2|$$

    ```m
    I = imread('test.bmp');
    I = double(I);
    w1 = [-1 0; 0 1];
    w2 = [0 -1; 1 0];
    % 这里需要以重复方式填充，或者采用镜像方式填充 'symmetric'
    G1 = imfilter(I, w1, 'corr', 'replicate');
    G2 = imfilter(I, w2, 'corr', 'replicate');
    G = abs(G1) + abs(G2);
    % 为了让 G 在图像中显示，需要添加参数 []，从而让图像灰度值变换到 0~255 之间
    figure, imshow(G, []);
    ```

* Sobel gradient

    $$
    w1 =
    \begin{pmatrix}
    -1 & -2 & -1 \\
    0 & 0 & 0 \\
    1 & 2 & 1
    \end{pmatrix}
    $$
    $$
    w2 =
    \begin{pmatrix}
    -1 & 0 & -1 \\
    -2 & 0 & -2 \\
    11 & 0 & -1
    \end{pmatrix}
    $$

    ```m
    I = imread('test.bmp');
    w1 = fspecial('sobel');
    w2 = w1';
    G1 = imfilter(I, w1);
    G2 = imfilter(I, w2);
    G = abs(G1) + abs(G2);
    ```

* Laplacian operator

    $$
    \nabla^2 f(x,y)=\frac{\partial^2 f}{\partial x^2}+\frac{\partial^2 f}{\partial y^2}$$
    $$
    \nabla^2 f=[f(i+1,j)+f(i-1,j)+f(i,j+1)+f(i,j-1)]-4f(i,j)
    $$
    Laplacian template is:
    $$
    w =
    \begin{pmatrix}
    0 & 1 & 0 \\
    1 & -4 & 1 \\
    0 & 1 & 0
    \end{pmatrix}
    $$

    ```m
    I = imread('test.bmp');
    I = double(I);
    w = [0 -1 0; -1 4 -1; 0 -1 0];
    L = imfilter(I, w1, 'corr', 'replicate');
    ```

    Laplacian 滤波器是各向同性的滤波器，类似的还有：
    $$
    w =
    \begin{pmatrix}
    1 & 1 & 1 \\
    1 & -8 & 1 \\
    1 & 1 & 1
    \end{pmatrix}
    $$
    $$
    w =
    \begin{pmatrix}
    1 & 4 & 1 \\
    4 & -20 & 4 \\
    1 & 4 & 1
    \end{pmatrix}
    $$

* 一阶和二阶导数算子的比较
    * 一阶导数通常会产生比较宽的边缘
    * 二阶倒数对于阶跃性边缘中心产生零交叉，对于屋顶状的边缘，二阶导数取极值 
    * 二阶导数对细节有较强的响应，对孤立噪声点也是如此

* Laplacian of a Gaussian (LoG)

    区分噪声和边缘是二阶导数算子的主要问题。
    $$h(r) = -\exp^{-\frac{r^2}{2\sigma^2}}$$
    $$\nabla^2 h(r)=-[\frac{r^2-\sigma^2}{\sigma^4}]\exp^{-\frac{r^2}{2\sigma^2}}$$
    其中 $h(r)$ 是高斯函数，下面是 LoG 滤波器。

    ```m
    I = imread('test.bmp');
    Id = double(I);
    h_log = fspecial('log', 5, 0.5);
    I_log = imfilter(Id, h_log, 'corr', 'replicate');
    ```

---

## 6 Image enhancement in frequency domain

