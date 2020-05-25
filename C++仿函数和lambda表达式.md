# SYCL开发者需要知道的C++特性：仿函数和lambda表达式

SYCL是基于C++11标准设计的。仿函数和lambda表达式这两项C++新特性在SYCL中被广泛用于定义kernel函数，
是开发者必须掌握的工具。这篇文章将帮助你了解这两个C++特性的基本运用。

## 仿函数 (Functor)
在软件开发过程中，我们会遇到一类情况，需要将函数作为参数传递给另外一个函数，比如使用std::sort对自定义的类型
进行排序。 C语言中的函数指针可以解决这个问题， 但也存在诸多问题。
C++ 为了进一步解决这个问题，增加了仿函数（又叫函数对象， Function Object）机制。

**仿函数的本质是一个重载了operator()的类实例。**
在实现上，仿函数利用运算符()，使我们可以像函数一样使用它。在下面的例子中，类MyFunctor的实例my_functor被用作仿函数：

```C++
#include <iostream>

class myFunctor
{
    public:
        myFunctor () {}
        int operator() (int x, int y) { return x + y; }
};

int main(){
    myFunctor my_functor;
    std::cout << my_functor(1,2);
    return 0;
}
```
在main函数中，对my_functor的调用很像普通函数，但本质上是对opertor()的调用。这也是“仿函数”这个译名的来源。
在本例中，我们调用的实际函数逻辑非常简单，仿函数并没有体现出任何优势。一种常见的对仿函数的运用是通过增加类的状态，
来实现相似的功能，比如：

```C++
#include <iostream>

class myFunctor
{   
    private:
        int base = 0;
    public:
        myFunctor (int x) : base(x) {}
        int operator() (int x, int y) { return x + y + base; }
};

int main(){
    myFunctor my_functor_1(2);
    myFunctor my_functor_2(10);
    std::cout << my_functor_1(1,2) << std::endl;
    std::cout << my_functor_2(1,2) << std::endl;
    return 0;
}
```
此例中，我们在myFunctor加入了成员变量base。在不同实例中，base的值不一样，因此其相应的仿函数行为也不一样。
这样只定义一个类，我们通过对不同实例传入不同参数，得到了不同的函数。  
下面我们再看一个常见的例子，使用仿函数定义排序中的比较函数：

```C++
#include <vector>
#include <iostream>
#include <algorithm>

struct Student {
  int age;
  Student(int x): age(x) {}
};

struct MyCompare { 
   bool operator()(const Student &a, const Student &b) {
       return a.age < b.age;
   }
};

int main(){
  std::vector<Student> vecStudent{{1}, {3}, {2}};
  MyCompare comparitor;
  std::sort(vecStudent.begin(), vecStudent.end(), comparitor);
  std::cout << vecStudent[1].age;
  return 0;
}
```

标准库函数std::sort提供了接口通过仿函数的方式传入自定义类型的比较函数。我们也可以在调用std::sort时
同时创建实例：
```C++
  // Simplify Code:
  // MyCompare comparitor;
  // std::sort(vecStudent.begin(), vecStudent.end(), comparitor);
  // To:
  std::sort(vecStudent.begin(), vecStudent.end(), MyCompare{});
```

## Lambda 表达式
Lambda 表达式机制是对仿函数在语法上的进一步的简化。Lambda表达式的返回值是一个仿函数。  
重新考察一下之前的排序代码例子，很多代码的存在只是为了语法的完整性，但并非我们实际所需。比如类MyCompare的定义，对运算符()重载的声明，以及对实例comparitor的创建。Lamdba表达式简化了非必要代码，之前的排序代码可以表达为：
```C++
// use lambda expression
std::sort(vecStudent.begin(), vecStudent.end(), [](const Student &a, const Student &b) { return a.x < b.y; });
// or 
auto comparitor = [](const Student &a, const Student &b) { return a.x < b.y; };
std::sort(vecStudent.begin(), vecStudent.end(), comparitor);
```
Lambda 表达式在创建和使用仅需要使用一次的函数时非常方便，相比之下仿函数更方便多次复用。

Lambda 表达式的基本形式是
```
[ captures ] (parameters) -> returnTypesDeclaration { lambdaStatements; }
```
其中：
* **\[captures\]** ：捕获列表，用于指定外部变量是如何传入（值传递，引用传递）lambda函数的。空列表[]表示lambda函数不会访问外部变量。捕获列表在声明lambda表达式时必需存在，不能省略。
* **(parameters)** : 参数列表，类似一般函数的参数列表，当参数数量为0时可以省略。
* **returnTypesDeclaration** ：返回值类型，在编译器可以推测出返回值类型时可以省略。 
* **{ lambdaStatemetns; }** : 表达式函数体，可以访问捕获列表和参数列表中的变量。

下面我们通过几个例子来进一步了解：
* 例1： 编译器自动推断返回类型
```C++
   auto comparitor = [](const Student &a, const Student &b) { return a.x < b.x; };
```
以上代码等同于
```C++
    auto comparitor = [](const Student &a, const Student &b) -> bool { return a.x < b.x; };
```
这里有一个细节，Lambda表达式的返回值的类型是不能直接指定并由编译器在编译时时生成。开发者只需要使用 auto 来指定其类型。

* 例2：捕获列表的基本使用
在这个例子中，我们将看到如何对外部变量以值传递或引用传递的方式进行捕获。
```C++
    int x = 1;
    int y = 2;
    int z = 3;
    auto func1 = [&](){};    // (1), default capture by reference
    auto func2 = [=](){};    // (2), default capture by copy
    auto func3 = [&,x](){};  // (3), default capture by reference, except x by copy
    auto func4 = [=,&x](){}; // (4), default capture by copy, excep x by reference
```
在例(1)和(2) 我们分别使用了&和=来指定**默认捕获**方式为**引用传递**和**值传递**。
默认捕获方式将作用于有所有没有显式指定传值方式的变量（包括this指针）。另外我们可以通过
&var或var 以**显示**得指定变量var传值方式为医用传递或值传递。在以上例子中：
1. func1 以引用传递的方式捕获x,y,z
2. func2 以值传递的方式捕获x,y,z
3. func3 以引用传递方式捕获y,z，以值传递方式捕获x
4. func4 以值传递方式捕获y,z, 以引用传递方式捕获y
捕获列表的存在简化Lambda表达式调用作用域内变量的过程。

## 总结

本文介绍了C++中仿函数和Lambda表达式的基本知识：
1. 仿函数是类重载了运算符()的实例，可以使用函数调用语法被调用。
2. Lambda表达式返回一个仿函数，可以用来简化使用仿函数的代码。
3. 相比于一般函数，Lambda表达式有捕获列表，灵活使用可以简化Lambda与作用域内变量的交互。





