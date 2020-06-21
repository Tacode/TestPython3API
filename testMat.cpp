#include <iostream>
#include <vector>
#include <map>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "conversion.h"

using namespace std;


map<string,int> LABEL_MAP = {{"__background__",0},{"aeroplane",1}, {"bicycle",2}, {"bird",3},
                             {"boat",4}, {"bottle",5},{"bus",6}, {"car",7}, {"cat",8}, {"chair",9},
                             {"cow",10}, {"diningtable",11}, {"dog",12}, {"horse",13},{"motorbike",14},
                             {"person",15}, {"pottedplant",16}, {"sheep",17}, {"sofa",18}, {"train",19},
                             {"tvmonitor",20}};
int testMat();        // 测试OpenCV图像传参及返回
int testMatFromTF();  // 测试图像传入及TensorFlow处理,使用了BlitzNet分割网络
int testMatFromTF2(); // 测试同时返回两个对象

int main(){
    testMat();
//    testMatFromTF();
//    testMatFromTF2();
    return 1;
}

int testMat(){

    Py_Initialize();
    if(!Py_IsInitialized()){
        cout << "[Error] Init error" << endl;
        return -1;
    }

    string change_dir = "sys.path.append('../scripts')";
    string model_dir = "../model";
    PyRun_SimpleString("import sys");
    PyRun_SimpleString(change_dir.c_str());

    PyObject *pModule = PyImport_ImportModule("testMat");

    if (pModule == nullptr){
        cout <<"[Error] Import module error" << endl;
        return -1;
    }

    cout << "[INFO] Get Module" << endl;

    PyObject *pClass = PyObject_GetAttrString(pModule, "TestMat");
    if (pClass == nullptr){
        cout << "[Error] Import class error" << endl;
        return -1;
    }
    cout << "[INFO] Get Class" << endl;

    PyObject *args1 = Py_BuildValue("(s)", model_dir.c_str());
    PyObject *pInstance = PyObject_Call(pClass,args1, nullptr); //创建实例
    assert(pInstance != nullptr);

    cv::Mat image = cv::imread("../air.jpg",CV_LOAD_IMAGE_UNCHANGED);
    NumpyAPI::NDArrayConverter *cvt = new NumpyAPI::NDArrayConverter();
    PyObject *pyImage = cvt->toNDArray(image.clone());
    assert(pyImage != nullptr);
    PyObject *pRetImage = PyObject_CallMethod(pInstance,
                                         "evaluate",
                                         "(O)",
                                         pyImage);
    if (pRetImage != nullptr){
        cv::Mat retImage = cvt->toMat(pRetImage);
        cv::imshow("image", retImage);
        cv::waitKey();
    }

    Py_DecRef(pModule);
    Py_DecRef(pClass);
    Py_DecRef(pInstance);
    Py_DecRef(args1);
    Py_Finalize();

    return 1;
}

int testMatFromTF(){
    Py_Initialize();
    if(!Py_IsInitialized()){
        cout << "[Error] Init error" << endl;
        return -1;
    }

    string change_dir = "sys.path.append('/home/tyl/PaperCode/Net')";//BlitzNet分割网络
    PyRun_SimpleString("import sys");
    PyRun_SimpleString(change_dir.c_str());

    PyObject *pModule = PyImport_ImportModule("BlitzNet");

    if (pModule == nullptr){
        cout <<"[Error] Import module error" << endl;
        return -1;
    }

    cout << "[INFO] Get Module" << endl;

    PyObject *pClass = PyObject_GetAttrString(pModule, "Evaluate");
    if (pClass == nullptr){
        cout << "[Error] Import class error" << endl;
        return -1;
    }
    cout << "[INFO] Get Class" << endl;

    PyObject *pInstance = PyObject_CallObject(pClass, nullptr); //创建实例
    if (pInstance == nullptr){
        cout << "[Error] Import Instance error" << endl;
        return -1;
    }
    cout << "[INFO] Get Instance" << endl;
    cout << "加载图片" << endl;
    cv::Mat image = cv::imread("../test.png",CV_LOAD_IMAGE_UNCHANGED);
    NumpyAPI::NDArrayConverter *cvt = new NumpyAPI::NDArrayConverter();
    PyObject *pyImage = cvt->toNDArray(image.clone());
    assert(pyImage != nullptr);
    cout << "调用类方法" << endl;
    PyObject *pDict = PyObject_CallMethod(pInstance,
                                          "GetDetection",
                                          "(O)",
                                          pyImage);
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

    PyObject *pRetImage = PyObject_CallMethod(pInstance,
                                         "GetSegmentation",
                                         "(O)",
                                         pyImage);
    if (pRetImage != nullptr){
        cv::Mat retImage = cvt->toMat(pRetImage);
        retImage.convertTo(retImage, CV_GRAY2RGB);
        cv::imshow("image", retImage);
        cv::waitKey();
    }

    Py_DecRef(pModule);
    Py_DecRef(pClass);
    Py_DecRef(pInstance);
    Py_Finalize();

    return 1;
}

int testMatFromTF2(){
    Py_Initialize();
    if(!Py_IsInitialized()){
        cout << "[Error] Init error" << endl;
        return -1;
    }

    string change_dir = "sys.path.append('/home/tyl/PaperCode/Net')";
    PyRun_SimpleString("import sys");
    PyRun_SimpleString(change_dir.c_str());
    PyRun_SimpleString("print(sys.path)");

    PyObject *pModule = PyImport_ImportModule("BlitzNet");

    if (pModule == nullptr){
        cout <<"[Error] Import module error" << endl;
        return -1;
    }

    cout << "[INFO] Get Module" << endl;

    PyObject *pClass = PyObject_GetAttrString(pModule, "Evaluate");
    if (pClass == nullptr){
        cout << "[Error] Import class error" << endl;
        return -1;
    }
    cout << "[INFO] Get Class" << endl;

    PyObject *pInstance = PyObject_CallObject(pClass, nullptr); //创建实例
    if (pInstance == nullptr){
        cout << "[Error] Import Instance error" << endl;
        return -1;
    }
    cout << "[INFO] Get Instance" << endl;
    cout << "加载图片" << endl;
    cv::Mat image = cv::imread("../test.png",CV_LOAD_IMAGE_UNCHANGED);
    NumpyAPI::NDArrayConverter *cvt = new NumpyAPI::NDArrayConverter();
    PyObject *pyImage = cvt->toNDArray(image.clone());
    assert(pyImage != nullptr);
    cout << "调用类方法" << endl;
    PyObject *pTuple = PyObject_CallMethod(pInstance,
                                          "GetDetectAndSeg",
                                          "(O)",
                                          pyImage);
    int tupleSize = PyTuple_Size(pTuple);
    assert(tupleSize == 2);
    PyObject *pDict = PyTuple_GetItem(pTuple, 0);
    PyObject *pRetImage = PyTuple_GetItem(pTuple, 1);

/*
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
    }*/

    //保存BlitzNet的目标检测结果
    map<int,vector<vector<int>> > loc_info;
    vector<int> coord;
    PyObject *pKeys = PyDict_Keys(pDict);
    for (Py_ssize_t i=0; i<PyDict_Size(pDict); i++){
        PyObject *key = PyList_GetItem(pKeys, i);
        string key_s = PyUnicode_AsUTF8(key);
        size_t pos = key_s.find("_", 0);
        string label = key_s.substr(0, pos);
        coord.clear();
        PyObject *pValue = PyDict_GetItem(pDict, key);

        for (Py_ssize_t j=0; j<PyList_Size(pValue); j++){
            PyObject *v = PyList_GetItem(pValue, j);
            if (PyLong_Check(v)){
                long v_l = PyLong_AsLong(v);
                coord.push_back(v_l);

            }
            else{
                cout << "Type Error" << endl;
                exit(0);
            }
        }
        loc_info[LABEL_MAP[label]].push_back(coord);

    }

    for (auto &entroy:loc_info){
        int k = entroy.first;
        vector<vector<int>> v = entroy.second;
        cout << k << ": ";
        for (auto &element:v){
            for_each(element.begin(), element.end(),[](int i){cout << i << " ";});
            cout << ", ";
        }
        cout << endl;
    }
    //保存BlitzNet的语义分割结果
    if (pRetImage != nullptr){
        cv::Mat retImage = cvt->toMat(pRetImage); // 分割结果
        retImage.convertTo(retImage, CV_GRAY2RGB);
        cv::imshow("image", retImage);
        cv::waitKey();
    }

    Py_DecRef(pModule);
    Py_DecRef(pClass);
    Py_DecRef(pInstance);
    Py_Finalize();

    return 1;
}
