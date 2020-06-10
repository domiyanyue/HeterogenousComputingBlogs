# SYCL 简介

欢迎来到我的SYCL教程！ SYCL 是一套使用单一源代码的基于OpenCL实现的异构平台编程框架。SYCL的发明使开发者可以完全使用C++语言开在异构计算系统中开发。
如果你对OpenCL很熟悉，你会发现SYCL中的概念与OpenCL十分相似，就可以把关注点放在SYCL的新特性上。对OpenCL不熟悉的学习者也不必担心，这部教程不需要任何OpenCL知识作为学习基础。

## 背景知识

异构计算系统是指使用多种处理器架构的计算系统，其处理器可能是通用处理器(CPU), 图形处理器(GPU), 可编程逻辑阵列(FPGA), 专用信号处理器(DSP)中的多种组合。近年来(2010 -)摩尔定理进展的减速使以通用处理器(CPU)为核心的计算平台出现了速度瓶颈，异构计算平台成为许多计算密集型热门应用的首选，比如机器学习，图像处理，自然语言处理等。

异构系统编程需要基于异构编程模型，常见异构编程模型包括包括OpenCL, CUDA, OpenACC等。OpenCL得益于它定义清晰的编程模型和易跨平台迁移的优点，得到了许多设备厂商的支持，拥有诸多用户。但是，OpenCL有3点重要缺陷：

1. 有限的C++语言支持。OpenCL 是基于C99的语言，开发者们无法在其中使用现代C++的特性。
2. 主机端(host)程序和设备端(device)程序的弱耦合机制让其开发过程十分容易出错。开发者需要使用两种不同语言进行开发，且不得不使用额外的脚本去生成住机端和设备端的共享代码。
3. OpenCL是一门低级语言(Low level Language), 需要开发者显示地表达主机端与设备端内存传输等低级操作，这使代码变得冗长。

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
1. **配置运行时环境** 为与设备端交互做准备的，比如创建代表具体设备的实例，创建与内存单元共享的内存等。在大部分框架中，运行时中的变量类型（设备，内存）以头文件内置类或者语言语法特性的方式提供。
2. **选择设备** 配置环境中的重要操作，开发者通过API指定程序执行的设备。
3. **创建与设备共享的内存** 设备中的数据需要从主机端程序中传入，当运算结束时，主机端需要从设备端拷贝结果。共享内存一般一类似数组的变量类型的方式存在。
4. **指定设备端代码** 在大部分编程模型中，开发者需要显示地指定在设备端执行的代码。
5. **提交设备端代码** 有了设备端代码，主机端还需要通知设备什么时候开始执行。一般的架构中，这个操作通过一个显示或隐式的队列完成，一组设备端代码会被"任务"的性质提交到主机端与设备端通信的队列，当设备端有足够的计算资源时，会获取并执行队列中任务。

### 设备端
设备端代码分成两部分：
1. **内存存取器定义** 在异构计算系统中，内存存取器(memory accessor)用于访问不同类别的内存。取决于编程模型和具体设备，设备端可以访问的内存种类很多，包括全局内存，局部内存，只读共享内存等。异构编程模型中，开发者往往需要显示地指定内存存取器的类型(访问内存类型，读写方式等)。
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

   default_selector device_selector;

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
下面我们来对代码中的基本结构进行详细解读：

### 引用SYCL头文件
```C++
#include <CL/sycl.hpp>

using namespace cl::sycl;
```

开发者需要引用 `CL/sycl.hpp` 来*配置运行时环境*。这个头文件包括了SYCL运行时需要的变量类型定义，包括queue, buffer, device 等。SYCL运行时类型都定义在命名空间 `cl::sycl `中。在本例中，为了代码的简介，我们加入了 `using namespace cl::sycl`.

### 选择设备(device)
```C++
default_selector device_selector;
```
这一行代码声明并初始化了**设备选择器(device selector)** 。设备选择器用于*选择设备*。SYCL内置了一些针对不同硬件的设备选择器，包括`cpu_selector`, `gpu_selector`, `host_selector` 和 `default_selector.` 除此之外，SYCL也支持开发者定制新的设备选择器来支持新的硬件。本例中，我们使用 `default_selector`，它表示SYCL运行时将自动决定使用的设备。


### 创建缓冲区(buffer)
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

这里需要注意的一点是buffer所在的作用域`{}`。在完整代码中`{`在buffer声名之前，`}`出现在打印结果前。作用域定义了`buffer`的存在区域(lifespan)。buffer在创建时被初始化，接管了vector中的数据。当代码执行到`}`时，buffer的析构函数(destructor)会自动将处理后的数据复制回`vec_a, vec_b, vec_c`中。
内存在主机端和设备端的转移是由buffer的构造函数和析构函数隐式的控制的。

### 构造命令组(command group)

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
命令组(command group)包含**内核函数(kernel function)** 和 **内存存取器定义(accessors definition)** 。在下文中我们将具体介。


### 创建命令队列(command queue)并提交命令组(command group)

```C++
...
    queue queue(device_selector);
    queue.submit([&] (handler& cgh) {
    ...
    }
```
上例展示了如何在SYCL中创建命令队列(command queue)并*提交设备端代码*。我们定义了一个于设备`device_selectr`关联的`queue`实例。设备端代码以命令组的形式被提交。SYCL中的命令队列提交是异步操作，这行代码在执行时会立即返回，命令组随后会在设备端运行。

### 存取器(accessor)

**存取器(accessor)** 是SYCL中用于访问缓冲区(buffer)的类型。设备端*内存存取器定义* 必须存在于命令组(command group)中。本例中，它们用来访问缓冲区中的全局内存(global memory)。

```C++
         auto a_acc = a_sycl.get_access<access::mode::read>(cgh);
         auto b_acc = b_sycl.get_access<access::mode::read>(cgh);
         auto c_acc = c_sycl.get_access<access::mode::write>(cgh);
```

`<缓冲区变量>.get_access` 将返回一个存取器。存取器有三个重要的属性：
1. **缓冲区**：可以访问的内存，在创建时指定。
2. **存取方式**：以模板参数的形式传入，常见的值包括read, write, read_write。存取方式有助于编译器优化内存访问。
3. **命令组handler**: 参数`cgh`表示存取器可以在这个命令组(command group)中的内核函数中(kernel function)使用。


### 内核函数(Kernel Function)

内核函数定义同样在命令组(command group)中：
```C++
         cgh.parallel_for<class VectorAdd>(range<1>(ArraySize), [=] (item<1> item) {
            c_acc[item] = a_acc[item] + b_acc[item];
         });
```
在SYCL中，内核函数由三部分组成：数据并行模型(data parallel model), 函数体(function body)以及内核名(kernel name).  
* 数据并行模型: 据并行模型由代码中的`parallel_for`和参数`range`的类型共同决定。本例中，数据并行模型是**基本数据并行模型(basic data parallel model)** ，这类模型将执行多个相互间不同步的 **工作项/线程(work-item/thread)** 。其他的数据并行模型包括**工作组数据并行(work-group data parallel)**, **单任务(single task)**, **等级制数据并行(hierarchical data parallel)**.
* 函数体: 函数体由仿函数(functor)形式表达，其参数列表由数据并行模型决定，本例中为`cl::sycl::item`。这个参数的类型和使用方法不是本文的重点，读者只需知道这个参数在每个线程中有不同的值，可以用作ID来访问不同数据。本例中我们从accessor`a_acc`和b`_acc`中读取数据，计算其和，并存储在`c_acc`中。
* 内核名：内核名称由`class VectorAdd` 定义。这里需要注意，内核名是一个类名，并需要在全局范围内声明。

## 总结

在这篇教程中，我们通过一个简单的向量加法程序初步了解了SYCL程序的基本组成：
* 将SYCL队列关联到指定设备。
* 使用缓冲区(buffer)隐式地控制主机端和设备端之间的内存传输。
* 在命令组(command group)定义存取器(accessor)和内核函数(kernel function)，通过队列提交到设备上运行。

这个例子同时体现了SYCL编程模型的以下优势：
* **支持C++语言**： 模板，lambda表达式等现代C++特性可以在SYCL程序中使用。SYCL运行时接口的设计（包括queue, buffer）是都资源获取即初始化（RAII）类型，
开发者不需要对资源的获取和释放进行显示的控制。
* **单一源代码**：SYCL主机端代码和设备端代码可以存在于同一文件并使用同种语言编写。开发者再无需使用不同编译器分别编译，项目的管理和部属也变得更加简洁。
* **隐式的数据传输**: 在SYCL中，主机端和设备端的内存传输时隐式控制的，开发者只需控制缓冲区(buffer)的变量域。SYCL运行时将自动控制内存中数据的转移。

如果这是你初次接触异构计算，本文中的新名词可能显得比较多且陌生。不必担心，在之后的教程中我会逐一讲解。敬请关注！
