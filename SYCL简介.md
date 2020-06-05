# SYCL 简介

欢迎来到我的SYCL教程！ SYCL 是一门使用单一源代码的基于OpenCL实现的异构平台编程框架。SYCL的发明使得开发者可以完全使用C++语言开发在异构系统中开发。
如果你对OpenCL很熟悉，你即将发现SYCL中的概念十分相似，可以把关注点放在SYCL的新特性上。对OpenCL不熟悉的学习者也不必担心，这部教程不需要OpenCL知识作为学习基础。
Welcome to my SYCL tutorial! SYCL is a single source heterogeneous programming model built on top of OpenCL that allows programmers to write heterogeneous application completely using C++. If you are familiar with OpenCL, the concepts in this tutorial should be familiar to you and you can focus on what's new in SYCL. If you are not, don't worry, this won't require any background knowledge in OpenCL. 

## 背景知识
## A Brief Background 

机构计算系统是指使用多种处理器架构的计算系统，其处理器组合可以是CPU + GPU, CPU + FPGA, CPU + DSP等。开发者对异构系统编程需要基于异构编程模型（包含编程语言），包括OpenCL, CUDA, OpenACC等。OpenCL在其中被广泛使用的，得益于它定义清晰的编程模型和易款平台迁移的优点。但是OpenCL有3点重要缺陷：
Heterogenous computing refers to systems that use more than one kind of processor (CPU + GPU, CPU + FPGA, CPU + DSP, etc.). To make programming for heterogeneous systems easy, people have come up with different programming models including OpenCL, CUDA, OpenACC, etc. 
OpenCL is one of the widely adopted one. It has a well-defined execution model that is portable across all types of devices. 
However, OpenCL received three major complaints:

1. 受限制的C++语言支持。OpenCL 是基于C99的语言，开发者们无法在其中使用现代C++的特性。
2. 主机端(host)程序和设备端(device)程序的弱耦合机制让OpenCL开发过程十分容易出错。开发者需要使用两种不同语言进行开发并且分别编译，通常，开发者会编写一些语言生成脚来去主机端和设备端的共享代码从而简化开发过程。
3. OpenCL是一门低级语言(Low level language), 需要开发者显示的表达主机端与设备端内存传输等低级操作。这使得代码变得冗长。
1. Limited support for C++. Developers do not benefit from new features in modern C++.  
2. The weak link between the host and device code is error-prone. Developers have to write in 2 different languages and compile host and device parts using different compilers. Often, users have to write their stringify script for purpose like code generation to automate the development process.
3. OpenCL is too verbose for many developers who don't want to explicitly write every low-level operation like memory transaction between host and device.

SYCL的设计在保留了OpenCL优点的同时，解决了OpenCL的以上问题：
1. 继承了OpenCL易迁移的执行模型。
2. 前端语言完全基于C++设计。
3. 使用了单一源代码的编程模型，开发者无需对主机端和设备端代码区别对待。
4. SYCL扩展了基于OpenCL的编程模型，让开发者可以用更高级的抽象概念编程。

SYCL was born reactive to OpenCL's pros and cons and aimed at a better heterogeneous framework.
1. It inherited the good execution model of OpenCL.
2. SYCL is purely based on C++. 
3. SYCL is a single source (no separation of device and host) programming model.
4. SYCL extends the programming model that allows developers to express at a high level of abstraction.            

## SYCL程序示例
## What Does SYCL Look Like?

在这部分中，我们将通过一个向量加法应用的例子来了解SYCL应用的结构。在学习中，请将重点放在程序的总体架构上而不要纠结于实例代码中包含的诸多SYCL语言细节。 下面是完整的代码：

I will lead you through a simple SYCL code sample performing vector add. This will give you an idea of the structure of a SYCL application. Please don't pay too much attention to details but focus on the higher level concepts. Here is the code:

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

Let's break it down to the basic build blocks of a SYCL application:

### 使用SYCL头文件
### Include Header Files
```C++
#include <CL/sycl.hpp>

using namespace cl::sycl;
```

SYCL 程序必须包含 `CL/sycl.hpp`。其中包括了SYCL运行时需要的变量类型定义，包括queue, buffer, device 等。SYCL运行时类型定于都在命名空间 `cl::sycl `中。在本例中，为了代码的简介，我们加入了 `using namespace cl::sycl`.

SYCL applications must include `CL/sycl.hpp` which contains APIs for SYCL runtime types like queue, buffer, device, etc. They all live under `cl::sycl` namespace. For the simplicity of this example, we put `using namespace` command at the beginning of the file.


### 选择设备(device)
### Select Device
```C++
default_selector device_selector;
```
这一行代码声明并初始化了**设备选择器(device selector)** 。设备选择器用于指定SYCL程序运行的硬件。SYCL内置了一些类型，其中包括`cpu_selector`, `host_selector`, `host_selector` 和 `default_selector.` SYCL支持开发者定制的设备选择器来支持不同的硬件。本例中，我们使用类型 `default_selector`，SYCL运行时将自动决定使用的设备。

This is how you specify the device (CPU, GPU, FPGA, etc) to execute on. SYCL provides a `default_selector` that will select an existing device in the system. SYCL also provides `cpu_selector`, `gpu_selector`, and allow you to customize your selector. In this example, we choose `default_selector` which let runtime picks for us.

### 创建缓冲区(buffer)
### Setup Buffers
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

In SYCL, a buffer is used to maintain an area of memory that can be shared between the host and one or more devices. Here we instantiate a **buffer type** with two *template parameters*: data type `float` and data dimension `1`. We also construct a **buffer instance** with two arguments: the first is the data source and the second one is the number of elements. SYCL provides interfaces for constructing buffers from different types of data sources like `std::vector` or `C arrays`.  
In the first line of this example, we create a 1-dimensional buffer object of containing element `float` of size `ArraySize` and initialized it with data in `vec_a`. 

Notice in the code there is a scope `{}` around buffers. This scope defines the life-span of buffer. When the buffer is constructed inside the scope, it automatically gets the ownership of the data. When the buffer goes out of scope, it copies data back to `vec_a`, `vec_b` and `vec_c`. The memory movement between host and device is handled implicitly in the buffer's constructor and destructor. 

### 构造命令组(command group)
### Create Command Group

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
在本例中，命令组(command group)包含两部分，

A command group is a single unit of work that will be executed on the device. You can see the command group
is passed as a functor (function object) parameter to to `submit` function. It accepts a parameter `handler` constructed by SYCL runtime which gives users the ability to access command group scope APIs. 
```C++
      queue.submit([&] (handler& cgh) { // start of command group
         // inputs and outputs accessor
         auto a_acc = a_sycl.get_access<access::mode::read>(cgh);
         auto b_acc = b_sycl.get_access<access::mode::read>(cgh);
         auto c_acc = c_sycl.get_access<access::mode::write>(cgh);
     
     // kernel function enqueue API `parallel_for`
         cgh.parallel_for<class VectorAdd>(range<1>(ArraySize), [=] (item<1> item) {
            c_acc[item] = a_acc[item] + b_acc[item];
         }); // end of command group
      });
```
In this case, the command group consists of a kernel function (defined by a kernel function enqueue API `parallel_for` which we will introduce later) and inputs and outputs defined by **accessor** object initialized by `get_access` API. 

### Construct Command Queue and Submit Command Group
```C++
...
    queue queue(device_selector);
    queue.submit([&] (handler& cgh) {
    ...
    }
```
A SYCL **queue** submits and triggers execution of **command group(s)** on a certain device. In this example, We first construct a
queue specifying the device it will submit to. Then we submit a command group to the device asynchronously. The submit command will return immediately and the execution of the command group will start later.

### Specify Accessors

**accessor** is the class to access buffer in SYCL. In this example, they are declared in command group to specify inputs and outputs from/to global memory.

```C++
         auto a_acc = a_sycl.get_access<access::mode::read>(cgh);
         auto b_acc = b_sycl.get_access<access::mode::read>(cgh);
         auto c_acc = c_sycl.get_access<access::mode::write>(cgh);
```
`<buffer>.get_access` returns accessor and gives it access to formerly created **buffer** objects `(a_sycl, b_sycl, c_sycl)`. For every accessor, there are three basic attributes:
1. **buffer**: memory it access which is determined when created. 
2. **access mode**: passed as template parameter. Typical values are read, write, read_write. This gives hints to the compiler to optimize the implementation. 
3. **command group handler**: the argument `cgh` indicates that the accessor will be available in kernel within this command group scope.

### Write Kernel Function
A kernel function is defined with 3 parts: data parallel model, kernel function body and kernel name. 
```C++
         cgh.parallel_for<class VectorAdd>(range<1>(ArraySize), [=] (item<1> item) {
            c_acc[item] = a_acc[item] + b_acc[item];
         });
```
In this example, data parallel model is defined by `parallel_for` API and `range` argument's type. Combined, they indicates the kernel will follow basic data parallel model which execute as multiple **work-items (threads)**. Other data parallel models are work-group data parallel, single task, and hierarchical data parallel.  

The kernel function body is encapsulated as a function object. It accepts an `item` as argument instructed by basic data parallel model. Inside the kernel function body, we add value from buffer `a_sycl` and `b_sycl` and save the result in `c_sycl` through accessors. Kernel argument `item` is **work-item id** (think of it as thread id) which tells the kernel that each execution of instance only add the value at **work-item id** position. Together, all threads will run in parallel to finish the work.

Kernel name is marked with `class VectorAdd`. Notice, it it a type and must be a class declared in global scope. 

## Summary

In this tutorial, we studied a "vector add" which includes basic elements of a SYCL program:
* Declare a SYCL queue object associated with certain devices.
* Use buffers to implicitly transfer ownership of memory between host and device.
* Write a command group that includes kernel function and accessors and submit it to the device.

We experienced SYCL's advantages as a programmnig model:
* **Support C++ language**: Developers can leverage C++ features to make the code more expressible. Also SYCL built-in APIs for queue, buffers are RAII types, which means developers has less bookkeeping job to control the life cycle of them.

* **Single Source**: SYCL host and device code can exist in the same file. You don't have to run 2 compiles separately on host and device anymore. This allows the project to be better organized and deployed.

* **Implicit Data Movement Support**: Unlike OpenCL and some other language where a lot of boilerplate code is needed for data
transfer between host and device, SYCL provides class "SYCL buffer" and runtime support to help automatically deduce memory movement.

The new terms introduced in this tutorial may be overwhelming at this point especially if you are new to heterogeneous computing. They will be covered in the following tutorials. Stay tuned! 


