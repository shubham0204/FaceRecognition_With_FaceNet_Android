package com.ml.shubham0204.facenetdetection

import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import com.ml.shubham0204.facenetdetection.ml.Annotator
import com.ml.shubham0204.facenetdetection.ml.FaceNetModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class AppViewModel: ViewModel() {

    var isDetectingFaces = MutableLiveData( false )

    var model = MutableLiveData<FaceNetModel>()
    var annotator = MutableLiveData<Annotator>()

    var showProgressOverlay = MutableLiveData( false )
    var progressOverlayMessage = MutableLiveData( "" )

    fun startProgressOverlay( message: String ) {
        progressOverlayMessage.value = message
        showProgressOverlay.value = true
    }
    suspend fun updateProgressOverlay(newMessage: String ) = withContext( Dispatchers.Main ) {
        progressOverlayMessage.value = newMessage
    }
    fun stopProgressOverlay() {
        showProgressOverlay.value = false
    }


}