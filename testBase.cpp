#include <iostream>
//#include <boost/python.hpp>
#include <string>
#include <cassert>
#include <map>
#include "/usr/include/python3.5/Python.h"
#include "unistd.h"
#include "malloc.h"
//#include "numpy/ndarrayobject.h"

using namespace std;
//using namespace boost::python;

int testFunction1(); // 调用Py函数，其中Python函数中调用其他函数
int testFunction2(); // 调用Py函数，其中Python函数中调用类
int testClass();     // 调用Py中类对象
int testMap();       // 测试返回类型为map

int main()
{
//    testFunction1();
//    testFunction2();
//    testClass();
    testMap();
    return 0;
}

int testFunction1(){
    Py_Initialize();
    if(!Py_IsInitialized()){
        cout << "[Error] Init error" << endl;
        return -1;
    }

    string change_dir = "sys.path.append('../scripts')";
    PyRun_SimpleString("import sys");
    PyRun_SimpleString(change_dir.c_str());

    PyObject *pModule = PyImport_ImportModule("testFunction");

    if (pModule == nullptr){
        cout <<"[Error] Import module error" << endl;
        return -1;
    }

    cout << "[INFO] Get Module" << endl;

    PyObject *pFunction = PyObject_GetAttrString(pModule, "func");
    if (pFunction == nullptr){
        cout << "[Error] Import function error" << endl;
        return -1;
    }

    cout << "[INFO] Get Function" << endl;
    PyObject *args = PyTuple_New(1);
    PyObject *args1 = PyUnicode_FromString("../air.jpg");

    PyTuple_SetItem(args, 0, args1);

    PyObject *pRet = PyObject_CallObject(pFunction, args);
    int res = 999;
    if (pRet){
        PyArg_Parse(pRet,"i", &res);
        cout << res << endl;
    }

    Py_DecRef(pModule);
    Py_DecRef(pFunction);
    Py_DecRef(args);
    Py_DecRef(args1);
    Py_DecRef(pRet);
    Py_Finalize();
    return 1;
}

int testFunction2(){
    Py_Initialize();
    if(!Py_IsInitialized()){
        cout << "[Error] Init error" << endl;
        return -1;
    }

    string change_dir = "sys.path.append('../scripts')";
    string model_dir = "../model";
    PyRun_SimpleString("import sys");
    PyRun_SimpleString(change_dir.c_str());

    PyObject *pModule = PyImport_ImportModule("testFunction2");

    if (pModule == nullptr){
        cout <<"[Error] Import module error" << endl;
        return -1;
    }

    cout << "[INFO] Get Module" << endl;

    PyObject *pFunction = PyObject_GetAttrString(pModule, "infer");
    if (pFunction == nullptr){
        cout << "[Error] Import function error" << endl;
        return -1;
    }
    cout << "[INFO] Get Function" << endl;
    PyObject *args = PyTuple_New(2);
    PyObject *args1 = PyUnicode_FromString(model_dir.c_str());
    PyObject *args2 = PyUnicode_FromString("../air.jpg");
    PyTuple_SetItem(args, 0, args1);
    PyTuple_SetItem(args, 1, args2);

    PyObject *pRet = PyObject_CallObject(pFunction,args);
    Py_DecRef(pModule);
    Py_DecRef(pFunction);
    Py_DecRef(args);
    Py_DecRef(args1);
    Py_DecRef(args2);
    Py_DecRef(pRet);
    Py_Finalize();
    return 1;
}

int testClass(){
    Py_Initialize();
    if(!Py_IsInitialized()){
        cout << "[Error] Init error" << endl;
        return -1;
    }

    string change_dir = "sys.path.append('../scripts')";
    string model_dir = "../model";
    PyRun_SimpleString("import sys");
    PyRun_SimpleString(change_dir.c_str());

    PyObject *pModule = PyImport_ImportModule("testClass");

    if (pModule == nullptr){
        cout <<"[Error] Import module error" << endl;
        return -1;
    }

    cout << "[INFO] Get Module" << endl;

    PyObject *pClass = PyObject_GetAttrString(pModule, "TestDemo");
    if (pClass == nullptr){
        cout << "[Error] Import class error" << endl;
        return -1;
    }
    cout << "[INFO] Get Class" << endl;
    PyObject *args1 = Py_BuildValue("(s)", model_dir.c_str());
    PyObject *pInstance = PyObject_Call(pClass,args1, nullptr); //创建实例
    assert(pInstance != nullptr);

    PyObject *args2 = Py_BuildValue("(s)", "../air.jpg");
    PyObject *pRet = PyObject_CallMethod(pInstance,"evaluate", "O", args2);

    if (pRet != nullptr){
        int res = 999;
        PyArg_Parse(pRet,"i", &res);
        cout << "成功返回参数: " << res << endl;
    }

    Py_DecRef(pModule);
    Py_DecRef(pClass);
    Py_DecRef(pInstance);
    Py_DecRef(args1);
    Py_DecRef(args2);
    Py_DecRef(pRet);
    Py_Finalize();
    return 1;

}

/*测试返回map类型*/
int testMap(){
    Py_Initialize();
    if(!Py_IsInitialized()){
        cout << "[Error] Init error" << endl;
        return -1;
    }

    string change_dir = "sys.path.append('../scripts')"; //路径相对于c++ bin文件的路径而言
    PyRun_SimpleString("import sys");
    PyRun_SimpleString(change_dir.c_str());

    PyObject *pModule = PyImport_ImportModule("testFunction");

    if (pModule == nullptr){
        cout <<"[Error] Import module error" << endl;
        return -1;
    }

    cout << "[INFO] Get Module" << endl;

    PyObject *pFunction = PyObject_GetAttrString(pModule, "func2");
    if (pFunction == nullptr){
        cout << "[Error] Import function error" << endl;
        return -1;
    }

    cout << "[INFO] Get Function" << endl;

    PyObject *pDict = PyObject_CallObject(pFunction, nullptr);
    PyObject *pKeys = PyDict_Keys(pDict);
    for (Py_ssize_t i=0; i<PyDict_Size(pDict); i++){
        PyObject *key = PyList_GetItem(pKeys, i);
        string key_s = PyUnicode_AsUTF8(key);
        cout << key_s << ": ";
        PyObject *pValue = PyDict_GetItem(pDict, key);

        for (Py_ssize_t j=0; j<PyList_Size(pValue); j++){
            PyObject *v = PyList_GetItem(pValue, j);
            if (PyLong_Check(v)){
                long v_l = PyLong_AsLong(v);
                cout << v_l << " " ;

            }else if(PyFloat_Check(v)){
                double v_d = PyFloat_AsDouble(v);
                cout << v_d << " ";
            }
        }
        cout << endl;
    }

    Py_DecRef(pModule);
    Py_DecRef(pFunction);
    Py_DecRef(pDict);
    Py_DecRef(pKeys);
    Py_Finalize();
    return 1;
}
