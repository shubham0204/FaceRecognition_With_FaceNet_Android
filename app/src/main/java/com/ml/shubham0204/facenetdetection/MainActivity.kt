package com.ml.shubham0204.facenetdetection

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.DocumentsContract
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.annotation.OptIn
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Cameraswitch
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.app.ActivityCompat
import androidx.documentfile.provider.DocumentFile
import com.ml.shubham0204.facenetdetection.ml.Annotator
import com.ml.shubham0204.facenetdetection.ml.FaceNetModel
import com.ml.shubham0204.facenetdetection.ml.Models
import com.ml.shubham0204.facenetdetection.ui.components.AppAlertDialog
import com.ml.shubham0204.facenetdetection.ui.components.DelayedVisibility
import com.ml.shubham0204.facenetdetection.ui.components.createAlertDialog
import com.ml.shubham0204.facenetdetection.ui.theme.AppTheme
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.ObjectInputStream
import java.io.ObjectOutputStream

class MainActivity : ComponentActivity() {

    // Serialized data will be stored ( in app's private storage ) with this filename.
    private val serializedDataFilename = "image_data"

    private val appViewModel by viewModels<AppViewModel>()
    private lateinit var fileReader : FileReader

    private val cameraPermissionStatus = mutableStateOf( false )
    private val cameraFacing = mutableIntStateOf( CameraSelector.LENS_FACING_BACK )

    // Use the device's GPU to perform faster computations.
    // Refer https://www.tensorflow.org/lite/performance/gpu
    private val useGpu = true

    // Use XNNPack to accelerate inference.
    // Refer https://blog.tensorflow.org/2020/07/accelerating-tensorflow-lite-xnnpack-integration.html
    private val useXNNPack = true

    // You may the change the models here.
    // Use the model configs in Models.kt
    // Default is Models.FACENET ;
    private val modelInfo = Models.FACENET

    private lateinit var processDirBlock: ((Uri?) -> Unit)


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            ActivityUI()
        }

        // We'll only require the CAMERA permission from the user.
        // For scoped storage, particularly for accessing documents, we won't require WRITE_EXTERNAL_STORAGE or
        // READ_EXTERNAL_STORAGE permissions. See https://developer.android.com/training/data-storage
        if ( ActivityCompat.checkSelfPermission( this , Manifest.permission.CAMERA ) != PackageManager.PERMISSION_GRANTED ) {
            // Request camera permission
            cameraPermissionLauncher.launch( Manifest.permission.CAMERA )
        }
        else {
            cameraPermissionStatus.value = true
        }

    }

    @Composable
    private fun ActivityUI() {
        AppTheme {
            Surface( modifier = Modifier
                .background(Color.White)
                .fillMaxSize() ) {
                Box {
                    Camera()
                    AppAlertDialog()
                    ProgressOverlay()
                    Controls()
                }
            }
        }
    }

    @Composable
    private fun Controls() {
        val isDetectingFaces by appViewModel.isDetectingFaces.observeAsState()
        Row {
            /*
            Button(onClick = { onNavigateToConfigureScreen() }) {
                Icon(imageVector = Icons.Default.Settings, contentDescription = "Configure")
                Text(text = "Configure")
            }*/
            Button(
                onClick = {
                    if ( isDetectingFaces == true ) {
                        appViewModel.isDetectingFaces.value = false
                    } else {
                        startDetection()
                    }} ,
                colors = ButtonDefaults.buttonColors( containerColor = if( isDetectingFaces == true ) { Color.Red } else { MaterialTheme.colorScheme.primaryContainer } )
            ) {
                Icon(imageVector = if( isDetectingFaces == true ) { Icons.Default.Stop } else { Icons.Default.Videocam }, contentDescription = "Start Detection")
                Text(text = if( isDetectingFaces == true ) { "Stop Detection" } else { "Start Detection" })
            }
            Button(
                onClick = {
                    if( cameraFacing.intValue == CameraSelector.LENS_FACING_BACK ) {
                        cameraFacing.intValue = CameraSelector.LENS_FACING_FRONT
                    }
                    else {
                        cameraFacing.intValue = CameraSelector.LENS_FACING_BACK
                    }
                } ,
                colors = ButtonDefaults.buttonColors( containerColor = if( cameraFacing.intValue == CameraSelector.LENS_FACING_FRONT ) { Color.Red } else { MaterialTheme.colorScheme.primaryContainer } )
            ) {
                Icon(imageVector = Icons.Default.Cameraswitch, contentDescription = "Switch Camera")
                Text(text = "Switch Camera" )
            }
        }
    }


    private val startDetection: ( () -> Unit ) = {
        val serializedDataFile = File( filesDir , serializedDataFilename )
        if ( !serializedDataFile.exists() ) {
            createAlertDialog(
                "Select Images Directory" ,
                "As mentioned in the project\'s README file, please select a directory which contains the images." ,
                "SELECT" ,
                "CANCEL" ,
                onPositiveButtonClick = {
                    startDetectionBySelectingImages()
                } ,
                onNegativeButtonClick = {
                }
            )
        }
        else {
            createAlertDialog(
                "Serialized Data",
                "Existing image data was found on this device. Would you like to load it?" ,
                "RESCAN" ,
                "LOAD" ,
                onPositiveButtonClick = {
                    startDetectionBySelectingImages()
                } ,
                onNegativeButtonClick = {
                    startDetectionWithStoredData()
                }
            )
        }
    }


    private fun startDetectionWithStoredData() {
        appViewModel.startProgressOverlay( "\uD83D\uDCF1 Initializing FaceNet ..." )
        CoroutineScope( Dispatchers.Default ).launch {
            val faceNetModel = FaceNetModel( this@MainActivity , modelInfo , useGpu , useXNNPack )
            mainThread { appViewModel.model.value = faceNetModel }
            fileReader = FileReader( faceNetModel )
            appViewModel.updateProgressOverlay( "\uD83D\uDDB4 Loading face data ..." )
            ioThread {
                val serializedDataFile = File( filesDir , serializedDataFilename )
                val objectInputStream = ObjectInputStream( FileInputStream( serializedDataFile ) )
                val faceList = objectInputStream.readObject() as ArrayList<Pair<String,FloatArray>>
                val annotator = Annotator()
                annotator.initialize( faceList )
                appViewModel.annotator.value = annotator
                objectInputStream.close()
            }
            mainThread {
                appViewModel.stopProgressOverlay()
            }
        }
        appViewModel.isDetectingFaces.value = true
    }


    private fun startDetectionBySelectingImages() {
        appViewModel.startProgressOverlay( "\uD83D\uDCF1 Initializing FaceNet ..." )
        CoroutineScope( Dispatchers.Default ).launch {
            val faceNetModel = FaceNetModel( this@MainActivity , modelInfo , useGpu , useXNNPack )
            mainThread { appViewModel.model.value = faceNetModel }
            fileReader = FileReader( faceNetModel )
        }
        launchChooseDirectoryIntent { dirUri ->
            CoroutineScope( Dispatchers.IO ).launch {
                appViewModel.updateProgressOverlay( "Scanning selected directory for images ..." )
                val childrenUri =
                    DocumentsContract.buildChildDocumentsUriUsingTree(
                        dirUri,
                        DocumentsContract.getTreeDocumentId( dirUri )
                    )
                val tree = DocumentFile.fromTreeUri(this@MainActivity , childrenUri)
                val images = ArrayList<Pair<String,Bitmap>>()
                var errorMessage = ""
                if ( tree!!.listFiles().isNotEmpty()) {
                    for ( doc in tree.listFiles() ) {
                        if ( doc.isDirectory && errorMessage.isEmpty() ) {
                            val name = doc.name!!
                            for ( imageDocFile in doc.listFiles() ) {
                                try {
                                    images.add( Pair( name , BitmapUtils.getFixedBitmap( this@MainActivity , imageDocFile.uri ) ) )
                                }
                                catch ( e : Exception ) {
                                    errorMessage = "Could not parse an image in $name directory. Make sure that the file structure is " +
                                            "as described in the README of the project and then restart the app."
                                    break
                                }
                            }
                        }
                        else {
                            errorMessage = "The selected folder should contain only directories. Make sure that the file structure is " +
                                    "as described in the README of the project and then restart the app."
                        }
                    }
                }
                else {
                    errorMessage = "The selected folder doesn't contain any directories. Make sure that the file structure is " +
                            "as described in the README of the project and then restart the app."
                }
                if ( errorMessage.isEmpty() ) {
                    appViewModel.updateProgressOverlay( "Reading images from selected directory ..." )
                    fileReader.run( images ) {
                        val serializedDataFile = File( filesDir , serializedDataFilename )
                        ObjectOutputStream( FileOutputStream( serializedDataFile )  ).apply {
                            writeObject( it.embeddedFaces )
                            flush()
                            close()
                        }
                        val annotator = Annotator()
                        annotator.initialize( it.embeddedFaces )
                        runBlocking( Dispatchers.Main ){
                            appViewModel.annotator.value = annotator
                        }
                    }
                }
                else {
                    createAlertDialog(
                        "Error while parsing directory" ,
                        "$errorMessage.\nPlease see the log below. Make sure that the file structure is " +
                                "as described in the README of the project and then tap RESELECT" ,
                        "RESELECT" ,
                        "CANCEL" ,
                        onPositiveButtonClick = {
                            startDetectionBySelectingImages()
                        } ,
                        onNegativeButtonClick = {
                            finish()
                        }
                    )
                }
                mainThread {
                    appViewModel.stopProgressOverlay()
                }
            }
        }
        appViewModel.isDetectingFaces.value = true
    }

    private suspend fun mainThread(block: (() -> Unit) ) {
        withContext( Dispatchers.Main ) {
            block()
        }
    }

    private suspend fun ioThread(block: () -> Unit) {
        withContext( Dispatchers.IO ) {
            block
        }
    }

    @OptIn(ExperimentalGetImage::class)
    @Composable
    private fun Camera() {
        val cameraPermissionStatus by remember{ cameraPermissionStatus }
        val cameraFacing by remember{ cameraFacing }
        val lifecycleOwner = LocalLifecycleOwner.current
        val context = LocalContext.current
        DelayedVisibility( cameraPermissionStatus ) {
            AndroidView(
                modifier = Modifier.fillMaxSize() ,
                factory = {
                    FaceDetectionOverlay(
                        lifecycleOwner ,
                        context ,
                        appViewModel
                    )
                } ,
                update = {
                    it.initializeCamera( cameraFacing )
                }
            )
        }
        DelayedVisibility( !cameraPermissionStatus ) {
            Box( modifier = Modifier.fillMaxSize() ) {
                Column( modifier = Modifier.align( Alignment.Center )) {
                    Text( text = "Allow Camera Permissions" )
                    Text( text = "The app cannot work without the camera permission." )
                    Button(
                        onClick = {
                            cameraPermissionLauncher.launch( Manifest.permission.CAMERA )
                                  } ,
                        modifier = Modifier.align( Alignment.CenterHorizontally )
                    ) {
                        Text(text = "Allow")
                    }
                }
            }
        }
    }


    @Composable
    private fun ProgressOverlay() {
        val show by appViewModel.showProgressOverlay.observeAsState()
        val message by appViewModel.progressOverlayMessage.observeAsState()
        DelayedVisibility( show ?: false ){
            Surface(
                color = Color.White.copy( alpha = 0.8f )
            ){
                Box( modifier = Modifier.fillMaxSize() ) {
                    Column(
                        verticalArrangement = Arrangement.Center ,
                        horizontalAlignment = Alignment.CenterHorizontally ,
                        modifier = Modifier.fillMaxSize() ) {
                        CircularProgressIndicator()
                        Spacer(modifier = Modifier.height(8.dp) )
                        Text( text = message ?: "" )
                    }
                }
            }
        }
    }

    private fun camaraPermissionDialog() {
        createAlertDialog(
            "Camera Permission" ,
            "The app couldn't function without the camera permission." ,
            "ALLOW" ,
            "CLOSE" ,
            onPositiveButtonClick = {
                cameraPermissionLauncher.launch( Manifest.permission.CAMERA )
            } ,
            onNegativeButtonClick = {
                finish()
            }
        )
    }


    private val cameraPermissionLauncher: ActivityResultLauncher<String> =
        registerForActivityResult( ActivityResultContracts.RequestPermission() ) { isGranted ->
            if ( isGranted ) { cameraPermissionStatus.value = true }
            else { camaraPermissionDialog() }
        }


    private val chooseDirectoryLauncher = registerForActivityResult( ActivityResultContracts.StartActivityForResult() ) {
        processDirBlock(it.data?.data)
    }


    private fun launchChooseDirectoryIntent( onResult: ((Uri?) -> Unit ) ) {
        this.processDirBlock = onResult
        chooseDirectoryLauncher.launch( Intent( Intent.ACTION_OPEN_DOCUMENT_TREE ) )
    }

}
