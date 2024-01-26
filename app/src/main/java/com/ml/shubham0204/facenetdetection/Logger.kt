package com.ml.shubham0204.facenetdetection

import android.util.Log

// Logs message using log_textview present in activity_main.xml
class Logger {

    companion object {

        fun log( message : String ) {
            Log.e( "APP" , message )
        }

    }

}