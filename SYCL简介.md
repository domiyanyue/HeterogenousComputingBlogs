# SYCL 简介

欢迎来到我的SYCL教程！ SYCL 是一套使用单一源代码的基于OpenCL实现的异构平台编程框架。SYCL的发明使开发者可以完全使用C++语言开在异构计算系统中开发。
如果你对OpenCL很熟悉，你会发现SYCL中的概念与OpenCL十分相似，可以把关注点放在SYCL的新特性上。对OpenCL不熟悉的学习者不必担心，这部教程不需要任何OpenCL知识作为学习基础。

## 背景知识

异构计算系统是指使用多种处理器架构的计算系统，其处理器可能是通用处理器(CPU), 图形处理器(GPU), 可编程逻辑门阵列(FPGA), 数字信号处理器(DSP)中的多种组合。近年来(2010 -)，摩尔定律的放缓使以通用处理器(CPU)为核心的计算平台出现了速度瓶颈，异构计算平台成为许多计算密集型热门应用的首选，比如机器学习，图像处理，自然语言处理，大数据处理等。

异构计算系统编程需要基于异构编程模型，常见异构编程模型包括包括OpenCL, CUDA, OpenACC等。OpenCL得益于它定义清晰的编程模型和易于跨平台迁移的优点，得到了绝大多数硬件厂商的支持，并且拥有诸多用户，形成了一定规模的生态。但是，OpenCL有三点重要缺陷：

1. 有限的C++语言支持。OpenCL 是基于C99的语言，开发者们无法在其中使用现代C++的特性。
2. 主机端(host)程序和设备端(device)程序的弱耦合机制让其开发过程十分容易出错。开发者需要使用两种不同语言进行开发，同时使用额外的脚本去生成主机端和设备端的共享代码。
3. OpenCL是一门低级语言(Low level Language), 开发者需要显示地表达主机端与设备端内存传输等低级操作，这使代码变得冗长且难以理解。

SYCL的设计在保留了OpenCL优点的同时，解决了以上问题：
1. SYCL继承了OpenCL易迁移的编程模型。
2. SYCL编程语言完全基于C++11并支持更高标准，开发者可以使用现代C++的特性。
3. SYCL使用了单一源代码的编程模型，无需对主机端和设备端代码区别对待。
4. SYCL扩展了基于OpenCL的编程模型，可以使用更高级的抽象概念(higher level abstraction)编程。

## 异构计算应用的基本结构
在进入SYCL程序样例之前，我们需要对异构计算应用的基本结构有所了解。这部分中，我们不针对任何特定编程模型，而是讲解普遍性适用于各种异构计算模型的概念。

异构计算应用由两部分组成：主机端和设备端。

### 主机端
在主机端程序中，除了完成正常CPU的运算，还需要与设备端程序进行交互，常见的操作有：
1. **配置运行时环境** 为与设备端交互做准备，比如创建代表具体设备的实例，创建内存共享单元等。在大部分框架中，运行时中的变量类型（设备，内存）以头文件的方式提供。
2. **选择设备** 配置环境中的重要操作，开发者指定程序执行的设备。
3. **创建与设备共享的内存** 设备中的数据需要从主机端程序中传入，当运算结束时，主机端需要从设备端拷贝结果。
4. **指定设备端代码** 开发者需要显示地指定哪些代码需要在设备端执行。
5. **提交设备端代码** 有了设备端代码，主机端还需要通知设备代码开始执行的时间。此操作可通过队列(queue)完成。

### 设备端
设备端代码分成两部分：
1. **内存存取器定义** 在异构计算系统中，内存存取器(memory accessor)用于访问不同类别的内存, 包括全局内存，局部内存，只读共享内存等。不同于常见的CPU程序，异构编程模型中，开发者需要显示地指定内存存取器的类型， 例如访问内存类别，读写方式等。
2. **内核函数** 运行在设备端的函数称作内核函数(kernel function)。

## SYCL程序示例

这部分中，我将通过一个向量加法的例子结合上一节内容来讲解SYCL程序的基本结构。在学习中，读者不必纠结于代码中的诸多SYCL语言细节，这些内容将在随后的教程中详细讲解。 以下是完整代码：

```C++
#include <iostream>
#include <vector>
#include <CL/sycl.hpp>

using namespace cl::sycl;
class VectorAdd;

int main() {
   const int ArraySize = 4;
   std::vector<float> vec_a{1.0f, 2.0f, 3.0f, 4.0f};
   std::vector<float> vec_b{5.0f, 6.0f, 7.0f, 8.0f};
   std::vector<float> vec_c(ArraySize);

   cpu_selector device_selector;

   queue queue(device_selector);
   std::cout << "Running on "
             << queue.get_device().get_info<info::device::name>()
             << "\n";
   {
      buffer<float, 1> a_sycl(vec_a.data(), ArraySize);
      buffer<float, 1> b_sycl(vec_b.data(), ArraySize);
      buffer<float, 1> c_sycl(vec_c.data(), ArraySize);
  
      queue.submit([&] (handler& cgh) {
         auto a_acc = a_sycl.get_access<access::mode::read>(cgh);
         auto b_acc = b_sycl.get_access<access::mode::read>(cgh);
         auto c_acc = c_sycl.get_access<access::mode::write>(cgh);

         cgh.parallel_for<class VectorAdd>(range<1>(ArraySize), [=] (item<1> item) {
            c_acc[item] = a_acc[item] + b_acc[item];
         });
      });
   }

   for(int i = 0; i < ArraySize; i++){
       std::cout << vec_a[i] << " + " << vec_b[i] <<  " = " << vec_c[i] << std::endl;
   }

   return 0;
}

```
下面我们将分别对主机端和设备端代码进行解读。

### 主机端代码

#### 引用SYCL头文件
```C++
#include <CL/sycl.hpp>

using namespace cl::sycl;
```

开发者需要引用 `CL/sycl.hpp` 来*配置运行时环境*。这个头文件包括了SYCL运行时需要的变量类型定义，包括queue, buffer, device 等。SYCL运行时类型都定义在命名空间 `cl::sycl `中。在本例中，为了代码的简洁，我们加入了 `using namespace cl::sycl`.

#### 选择设备(device)
```C++
cpu_selector device_selector;
```
这一行代码声明并初始化了**设备选择器(device selector)** 。SYCL内置了一些针对不同硬件的设备选择器，包括`cpu_selector`, `gpu_selector`, `host_selector` 和 `default_selector.` 除此之外，SYCL也支持开发者定制新的设备选择器来支持新的硬件。本例中，我们使用 `cpu_selector`，它表示SYCL运行时将使用CPU执行设备端代码。

#### 创建缓冲区(buffer)
```C++
{
      buffer<float, 1> a_sycl(vec_a.data(), ArraySize);
      buffer<float, 1> b_sycl(vec_b.data(), ArraySize);
      buffer<float, 1> c_sycl(vec_c.data(), ArraySize);
      . . .
}
```
这段代码展示了在SYCL中*创建与设备共享的内存*。 缓冲区(buffer)是SYCL中的一个重要类型，用来表示在主机端(host)和设备端(device)间共享的内存。本例中，我们使用两个参数实例化了模板类buffer: 变量类型 `float` 和数据维度 `1` 。在buffer构造函数中，我们传入了数据源和数据量(ArraySize)。buffer类型支持直接从`std::vector`或`C数组`中传入数据。
这段代码的第一行，我们创建了一个一维浮点数大小为`ArraySize`的缓冲区，并用`vec_a`中的数据进行了初始化。

这里需要注意的一点是buffer所在的作用域`{}`。在完整代码中，buffer的`{}`出现在两个`cout`之间。作用域定义了`buffer`的存在区域(lifespan)。buffer在创建时被初始化，接管了vector中的数据。当代码执行到`}`时，buffer的析构函数(destructor)会自动将处理后的数据复制回`vec_a, vec_b, vec_c`中。
内存在主机端和设备端的转移是由buffer的构造函数和析构函数隐式地控制的。

#### 构造命令组(command group)

命令组(command group)用于*指定设备端代码*，SYCL中设备端运行代码必须写在命令组中。在本例中，命令组以仿函数 (functor)的形式作为参数传入`queue.submit`函数。命令组的仿函数需要接受参数`handler`，`handler` 由SYCL运行时创建，开发者将使用他来访问命令组(command group)中的程序接口(API)。

```C++
      queue.submit([&] (handler& cgh) { // 指令组 (command group) 开始
         // inputs and outputs accessor
         auto a_acc = a_sycl.get_access<access::mode::read>(cgh);
         auto b_acc = b_sycl.get_access<access::mode::read>(cgh);
         auto c_acc = c_sycl.get_access<access::mode::write>(cgh);

         // kernel function enqueue API `parallel_for`
         cgh.parallel_for<class VectorAdd>(range<1>(ArraySize), [=] (item<1> item) {
            c_acc[item] = a_acc[item] + b_acc[item];
         }); // 指令组 (command group)结束
      });
```
命令组(command group)包含**内核函数(kernel function)** 和 **内存存取器定义(accessors definition)** ，在下文中我们将具体介。

#### 创建命令队列(command queue)并提交命令组(command group)

```C++
...
    queue queue(device_selector);
    queue.submit([&] (handler& cgh) {
    ...
    }
```
上例展示了如何在SYCL中创建命令队列(command queue)并*提交设备端代码*。我们定义了一个与设备`device_selectr`关联的`queue`实例，设备端代码以命令组的形式被提交。SYCL中的命令队列提交是异步操作，`queue.submit`这行代码在执行时会立即返回，命令组随后会在设备端运行。

### 设备端代码
设备端代码全部存在于命令组中。

#### 存取器(accessor)

存取器是SYCL中用于访问缓冲区(buffer)的类型。本例中，它们用来访问缓冲区中的全局内存(global memory)。

```C++
         auto a_acc = a_sycl.get_access<access::mode::read>(cgh);
         auto b_acc = b_sycl.get_access<access::mode::read>(cgh);
         auto c_acc = c_sycl.get_access<access::mode::write>(cgh);
```

buffer成员函数 `get_access` (例如`a_sycl.get_access`) 将返回存取器。存取器有三个重要的属性, 我们以第一行代码 `auto a_acc = a_sycl.get_access<access::mode::read>(cgh)` 为例来解读：
1. **缓冲区 (a_sycl)** ：可以访问的内存，在创建时指定。
2. **存取方式 (access::mode::read)** ：以模板参数的形式传入，常见的值包括`read`, `write`, `read_write`。存取方式可以帮助编译器优化内存访问速度。
3. **命令组handler (cgh)** : 参数`cgh`表示存取器可以在这个命令组(command group)中的内核函数中(kernel function)使用。


#### 内核函数(Kernel Function)

内核函数定义同样在命令组(command group)中：
```C++
         cgh.parallel_for<class VectorAdd>(range<1>(ArraySize), [=] (item<1> item) {
            c_acc[item] = a_acc[item] + b_acc[item];
         });
```
内核函数是运行在设备端的函数。常见的编程语言中，函数由两部分组成函数名(function name)和函数体(function body)。
在SYCL中，内核函数由三部分组成：**数据并行模型(data parallel model)**, **函数体(function body)** 以及**内核名(kernel name)**.
* 数据并行模型: 数据并行模型定义了该函数将以何种并行方式在设备端运行。数据并行模型由代码中的`cgh.parallel_for<class VectorAdd>(range<1>(ArraySize)` 决定，本例中的数据并行模型是**基本数据并行模型(basic data parallel model)** , 多个线程(thread)将并行的执行函数体中的代码。其它数据并行模型包括**工作组数据并行(work-group data parallel)**, **单任务(single task)**, **等级制数据并行(hierarchical data parallel)**.

* 函数体: 函数体由仿函数(functor)形式表达 `[=] (item<1> item) {c_acc[item] = a_acc[item] + b_acc[item];}`。 其参数列表`(item<1> item)`由数据并行模型决定。读者这里只需知道这个参数`item`在每个线程中有不同的值，可以用在不同线程中访问不同地址的数据就足够了。本例中，SYCL运行时将启动`ArraySize`个线程，每个线程中的`item`值分别是`0, 1, 2.. ArraySize-1`，每个线程将完成向量中一个单元的加法。`ArraySize`个线程共同完成了向量加法。关于函数体执行的细节我们在随后的教程中将详细解读。

* 内核名：内核名称由`class VectorAdd` 定义。这里需要注意，内核名是一个类名，并需要在全局范围内声明。

## 总结

在这篇教程中，我们通过一个简单的向量加法程序初步了解了SYCL程序的基本组成：
* 将SYCL队列关联到指定设备。
* 使用缓冲区(buffer)隐式地控制主机端和设备端之间的内存传输。
* 在命令组(command group)定义存取器(accessor)和内核函数(kernel function)，通过队列提交到设备上运行。

这个例子同时体现了SYCL编程模型的以下优势：
* **支持C++语言**： 模板，lambda表达式等现代C++特性可以在SYCL程序中使用。SYCL运行时接口的设计（包括queue, buffer）是都资源获取即初始化（RAII）类型，开发者不需要对资源的获取和释放进行显示的控制，减少了冗长的代码以及将降低了出错的风险。
* **单一源代码**：SYCL主机端代码和设备端代码可以存在于同一文件并使用同一种语言编写。开发者再无需使用不同编译器分别编译，项目的管理和布属变得更加简洁易用。
* **隐式的数据传输**: 在SYCL中，主机端和设备端的内存传输是隐式控制的，开发者只需控制缓冲区(buffer)的作用域，SYCL运行时将自动控制内存中数据的转移。

如果这是你初次接触异构计算，本文中的新名词可能显得比较多且陌生。不必担心，在之后的教程中我们会逐一讲解。敬请关注！
