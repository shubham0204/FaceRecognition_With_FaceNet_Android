package com.ml.shubham0204.facenetdetection

import android.graphics.RectF

data class Prediction( var bbox : RectF, var label : String , var maskLabel : String = "" )