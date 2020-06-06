# SYCL 简介

欢迎来到我的SYCL教程！ SYCL 是一套使用单一源代码的基于OpenCL实现的异构平台编程框架。SYCL的发明使开发者可以完全使用C++语言开在异构计算系统中开发。
如果你对OpenCL很熟悉，你即将发现SYCL中的概念十分相似，并可以把关注点放在SYCL的新特性上。对OpenCL不熟悉的学习者也不必担心，这部教程不需要任何OpenCL知识作为学习基础。

## 背景知识

机构计算系统是指使用多种处理器架构的计算系统，其处理器组合可能是通用处理器(CPU), 图形处理器(GPU), 可编程逻辑阵列(FPGA), 专用信号处理器(DSP)中的多种组合。近年来(2010 -)摩尔定理进展的减速使以通用处理器(CPU)为核心的计算平台出现了速度瓶颈，异构计算平台成为许多计算密集型热门应用比如机器学习，图像处理，自然语言处理等的首选。

异构系统编程需要基于异构编程模型，常见计算模型包括包括OpenCL, CUDA, OpenACC等。OpenCL得益于它明确定义的编程模型和易跨平台迁移的优点，得到了许多设备厂商的支持，拥有诸多用户。但是OpenCL有3点重要缺陷：

1. 受限制的C++语言支持。OpenCL 是基于C99的语言，开发者们无法在其中使用现代C++的特性。
2. 主机端(host)程序和设备端(device)程序的弱耦合机制让其开发过程十分容易出错。开发者需要使用两种不同语言进行开发并分别编译。在OpenCL开发环境中，开发者会编写一些字宗语言生成脚来去主机端和设备端的共享代码从而简化编程过程。
3. OpenCL是一门低级语言(Low level Language), 需要开发者显示地表达主机端与设备端内存传输等低级操作，这使得代码变得冗长。

SYCL的设计在保留了OpenCL优点的同时，解决了OpenCL的以上问题：
1. 继承了OpenCL易迁移的执行模型。
2. 前端语言完全基于C++设计。
3. 使用了单一源代码的编程模型，开发者无需对主机端和设备端代码区别对待。
4. SYCL扩展了基于OpenCL的编程模型，让开发者可以用更高级的抽象概念编程。         

## SYCL程序示例

在这部分中，我们将通过一个向量加法应用的例子来了解SYCL应用的结构。在学习中，请将重点放在程序的总体架构上而不要纠结于实例代码中包含的诸多SYCL语言细节。 下面是完整的代码：

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
下面我们会对代码中的基本单元进行一一解读：

### 使用SYCL头文件
```C++
#include <CL/sycl.hpp>

using namespace cl::sycl;
```

SYCL 程序必须包含 `CL/sycl.hpp`。其中包括了SYCL运行时需要的变量类型定义，包括queue, buffer, device 等。SYCL运行时类型定于都在命名空间 `cl::sycl `中。在本例中，为了代码的简介，我们加入了 `using namespace cl::sycl`.

### 选择设备(device)
```C++
default_selector device_selector;
```
这一行代码声明并初始化了**设备选择器(device selector)** 。设备选择器用于指定SYCL程序运行的硬件。SYCL内置了一些类型，其中包括`cpu_selector`, `host_selector`, `host_selector` 和 `default_selector.` SYCL支持开发者定制的设备选择器来支持不同的硬件。本例中，我们使用类型 `default_selector`，SYCL运行时将自动决定使用的设备。


### 创建缓冲区(buffer)
```C++
{
      buffer<float, 1> a_sycl(vec_a.data(), ArraySize);
      buffer<float, 1> b_sycl(vec_b.data(), ArraySize);
      buffer<float, 1> c_sycl(vec_c.data(), ArraySize);
      . . .
}
```
缓冲区(buffer)是SYCL引入的类型，用来表示在主机端(host)和设备端(device)间共享的内存。本例中，我们使用两个参数实例化了模板类buffer: 变量类型 `float` 和数据维度 `1` 。在buffer构造函数中，我们传入了数据源和数据量(ArraySize)。buffer类型支持直接从`std::vector`或`C数组`中传入数据。
这段代码的第一行，我们创建了一个一维浮点数大小为`ArraySize`的缓冲区，并用`vec_a`中的数据进行了初始化。

这里需要注意的一点是buffer所在的作用域`{}`。在完整代码中`{`在buffer声名之前，`}`出现在打印结果前。作用域定义了`buffer`的存在区域(lifespan)。buffer在创建时被初始化，接管了vector中的数据。当代码执行到`}`时，buffer的析构函数(destructor)会自动将处理后的数据复制回`vec_a, vec_b, vec_c`中。
内存在主机端和设备端的转移是有buffer的构造函数和析构函数隐式的控制的。

### 构造命令组(command group)

命令组(command group)是一组在设备端运行的代码，在本例中，指令组以仿函数 (functor)的形式传入`submit`函数。指令组的仿函数需要接受参数`handler`，`handler` 由SYCL运行时创建，开发者将使用他来访问命令组(command group)中的程序接口(API)。

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
命令组(command group)包含两部分：
* **内核函数(kernel function)** : 内核函数以仿函数(functor)的形式存在于enqueue API `parallel_for`中。在下文中我们将介绍函数的组成。
* **输入和输出(inputs and outputs)** ：内核函数的输入输出由**存取器(accessor)** 定义，存取器在内核函数中用于访问内存。下文中将单独介绍存取器。


### 创建命令队列(command queue)并提交命令组(command group)

```C++
...
    queue queue(device_selector);
    queue.submit([&] (handler& cgh) {
    ...
    }
```
命令队列(command queue)在SYCL中用于向设备端提交命令组(command group)。上例中，我们定义了一个`queue`实例并传入`device_selectr`作为设备参数。
下一行中，指令组被提交(submit)，SYCL中的命令队列提交是异步操作，这行代码在执行时会立即返回，命令组随后会在设备端运行。

### 使用存取器(accessor)

**存取器(accessor)** 是SYCL中专门用于访问缓冲区(buffer)的类型。本例中，它们在命令组(command group)中使用来访问缓冲区中的全局内存(global memory)。

```C++
         auto a_acc = a_sycl.get_access<access::mode::read>(cgh);
         auto b_acc = b_sycl.get_access<access::mode::read>(cgh);
         auto c_acc = c_sycl.get_access<access::mode::write>(cgh);
```

获取访问某个缓冲区的常用方式是`<缓冲区变量>.get_access` ，该函数将返回一个存取器。存取器有三个重要的属性：
1. **缓冲区**：可以访问的内存，在创建时指定。
2. **存取方式**：以模板参数的形式传入，常见的值包括read, write, read_write。存取方式有助于编译器优化内存访问。
3. **命令组handler**: 参数`cgh`表示存取器可以在这个命令组(command group)中的内核函数中(kernel function)使用。


### 实现内核函数(Kernel Function)

内核函数由三部分组成：数据并行模型(data parallel model), 函数体(function body)以及内核名(kernel name).  
A kernel function is defined with 3 parts: data parallel model, kernel function body and kernel name. 
```C++
         cgh.parallel_for<class VectorAdd>(range<1>(ArraySize), [=] (item<1> item) {
            c_acc[item] = a_acc[item] + b_acc[item];
         });
```

* 数据并行模型: 据并行模型由代码中的`parallel_for`和参数`range`的类型共同决定。本例中，数据并行模型是**基本数据并行模型(basic data parallel model)** ，这类模型将执行多个相互间不同部的 **工作项/线程(work-item/thread)** 。其他的数据并行模型包括**工作组数据并行(work-group data parallel)**, **单任务(single task)**, **等级制数据并行(hierarchical data parallel)**.

* 函数体: 函数体由仿函数(functor)形式表达，

* 内核名：内核名称由`class VectorAdd` 定义。这里需要注意，内核名是一个类名，并需要在全局范围内声明。

## 总结

在这篇教程中，我们通过一个简单的向量加法程序初步了解了SYCL程序的基本组成：
* 将SYCL队列关联到指定设备。
* 使用缓冲区(buffer)隐式地控制主机端和设备端之间的内存传输。
* 实现一个包括内核函数(kernel function)的命令组(command group)，通过队列提交到设备上运行。

这个例子同时体现了SYCL编程模型的一下势：
* **支持C++语言**： 模板，lambda表达式等现代C++特性可以在SYCL程序中使用。SYCL运行时接口的设计（包括queue, buffer）是都资源获取即初始化（RAII）类型，
开发者不需要对资源的获取和释放进行显示的控制。
* **单一源代码**：SYCL主机端代码和设备端代码可以存在于同一文件并使用同种语言编写。开发者再无需使用不同编译器分别编译，项目的管理和部属也变得更加简洁。
* **隐式的数据传输**: 在SYCL中，主机端和设备端的内存传输时隐式控制的，开发者只需控制缓冲区(buffer)的变量域。SYCL运行时将自动控制内存中数据的转移。

如果这是你初次接触异构计算，本文中的新名词可能显得比较多且陌生。不必担心，在之后的教程中我会逐一讲解。敬请关注！
