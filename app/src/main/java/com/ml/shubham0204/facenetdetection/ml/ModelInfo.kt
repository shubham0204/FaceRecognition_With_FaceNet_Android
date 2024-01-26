package com.ml.shubham0204.facenetdetection.ml

class ModelInfo(
    val name : String ,
    val assetsFilename : String ,
    val cosineThreshold : Float ,
    val l2Threshold : Float ,
    val outputDims : Int ,
    val inputDims : Int ,
    val description: String = "" )