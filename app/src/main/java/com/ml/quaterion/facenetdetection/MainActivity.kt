/*
 * Copyright 2021 Shubham Panchal
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.ml.quaterion.facenetdetection

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.ExifInterface
import android.net.Uri
import android.os.Bundle
import android.provider.DocumentsContract
import android.text.method.ScrollingMovementMethod
import android.util.Size
import android.view.Window
import android.view.WindowManager
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.documentfile.provider.DocumentFile
import androidx.lifecycle.LifecycleOwner
import com.google.common.util.concurrent.ListenableFuture
import java.io.*
import java.util.concurrent.Executors


class MainActivity : AppCompatActivity() {

    private val REQUEST_CAMERA_PERMISSION = 101
    private val REQUEST_DIRECTORY_ACCESS  = 102
    private var isSerializedDataStored = false

    // Serialized data will be stored ( in app's private storage ) with this filename.
    private val SERIALIZED_DATA_FILENAME = "image_data"

    // Shared Pref key to check if the data was stored.
    private val SHARED_PREF_IS_DATA_STORED_KEY = "is_data_stored"

    private lateinit var previewView : PreviewView
    private lateinit var frameAnalyser  : FrameAnalyser
    private lateinit var model : FaceNetModel
    private lateinit var fileReader : FileReader
    private lateinit var cameraProviderFuture : ListenableFuture<ProcessCameraProvider>
    private lateinit var sharedPreferences: SharedPreferences


    companion object {

        lateinit var logTextView : TextView

        fun setMessage( message : String ) {
            logTextView.text = message
        }

    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Remove the Status Bar for a fullscreen display
        // See this SO answer -> https://stackoverflow.com/a/5436403/10878733
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        this.window.setFlags( WindowManager.LayoutParams.FLAG_FULLSCREEN,WindowManager.LayoutParams.FLAG_FULLSCREEN );
        setContentView(R.layout.activity_main)

        // Implementation of CameraX preview

        previewView = findViewById( R.id.preview_view )
        logTextView = findViewById( R.id.log_textview )
        logTextView.movementMethod = ScrollingMovementMethod()
        // Necessary to keep the Overlay above the PreviewView so that the boxes are visible.
        val boundingBoxOverlay = findViewById<BoundingBoxOverlay>( R.id.bbox_overlay )
        boundingBoxOverlay.setWillNotDraw( false )
        boundingBoxOverlay.setZOrderOnTop( true )

        frameAnalyser = FrameAnalyser( this , boundingBoxOverlay)
        model = FaceNetModel( this )
        fileReader = FileReader( this )


        // We'll only require the CAMERA permission from the user.
        // For scoped storage, particularly for accessing documents, we won't require WRITE_EXTERNAL_STORAGE or
        // READ_EXTERNAL_STORAGE permissions. See https://developer.android.com/training/data-storage
        if ( ActivityCompat.checkSelfPermission( this , Manifest.permission.CAMERA ) != PackageManager.PERMISSION_GRANTED ) {
            ActivityCompat.requestPermissions( this , arrayOf( Manifest.permission.CAMERA ) , REQUEST_CAMERA_PERMISSION )
        }
        else {
            startCameraPreview()
        }

        sharedPreferences = getSharedPreferences( getString( R.string.app_name ) , Context.MODE_PRIVATE )
        isSerializedDataStored = sharedPreferences.getBoolean( SHARED_PREF_IS_DATA_STORED_KEY , false )
        if ( !isSerializedDataStored ) {
            Logger.log( "No serialized data was found. Select the images directory.")
            showSelectDirectoryDialog()
        }
        else {
            val alertDialog = AlertDialog.Builder( this ).apply {
                setTitle( "Serialized Data")
                setMessage( "Existing image data was found on this device. Would you like to load it?" )
                setCancelable( false )
                setNegativeButton( "LOAD") { dialog, which ->
                    dialog.dismiss()
                    frameAnalyser.faceList = loadSerializedImageData()
                    Logger.log( "Serialized data loaded.")
                }
                setPositiveButton( "RESCAN") { dialog, which ->
                    dialog.dismiss()
                    launchChooseDirectoryIntent()
                }
                create()
            }
            alertDialog.show()
        }

    }

    // ---------------------------------------------- //

    // Attach the camera stream to the PreviewView.
    private fun startCameraPreview() {
        cameraProviderFuture = ProcessCameraProvider.getInstance( this )
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            bindPreview(cameraProvider) },
            ContextCompat.getMainExecutor(this) )
    }

    private fun bindPreview(cameraProvider : ProcessCameraProvider) {
        val preview : Preview = Preview.Builder().build()
        val cameraSelector : CameraSelector = CameraSelector.Builder()
            .requireLensFacing( CameraSelector.LENS_FACING_FRONT )
            .build()
        preview.setSurfaceProvider( previewView.surfaceProvider )
        val imageFrameAnalysis = ImageAnalysis.Builder()
            .setTargetResolution(Size( 480, 640 ) )
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
        imageFrameAnalysis.setAnalyzer(Executors.newSingleThreadExecutor(), frameAnalyser )
        cameraProvider.bindToLifecycle(this as LifecycleOwner, cameraSelector, preview , imageFrameAnalysis  )
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        if ( requestCode == REQUEST_CAMERA_PERMISSION ) {
            if ( grantResults[ 0 ] == PackageManager.PERMISSION_GRANTED ) {
                startCameraPreview()
            }
            else if ( grantResults[ 0 ] == PackageManager.PERMISSION_DENIED ) {
                val alertDialog = AlertDialog.Builder( this ).apply {
                    setTitle( "Camera Permission")
                    setMessage( "The app couldn't function without the camera permission." )
                    setCancelable( false )
                    setPositiveButton( "ALLOW" ) { dialog, which ->
                        dialog.dismiss()
                        ActivityCompat.requestPermissions( context as MainActivity ,
                            arrayOf( Manifest.permission.CAMERA ) ,
                            REQUEST_CAMERA_PERMISSION )
                    }
                    setNegativeButton( "CLOSE" ) { dialog, which ->
                        dialog.dismiss()
                        finish()
                    }
                    create()
                }
                alertDialog.show()
            }
        }
    }

    // ---------------------------------------------- //

    // Open File chooser to choose the images directory.
    private fun showSelectDirectoryDialog() {
        val alertDialog = AlertDialog.Builder( this ).apply {
            setTitle( "Select Images Directory")
            setMessage( "As mentioned in the project\'s README file, please select a directory which contains the images." )
            setCancelable( false )
            setPositiveButton( "SELECT") { dialog, which ->
                dialog.dismiss()
                launchChooseDirectoryIntent()
            }
            create()
        }
        alertDialog.show()
    }

    private fun launchChooseDirectoryIntent() {
        val intent = Intent( Intent.ACTION_OPEN_DOCUMENT_TREE )
        startActivityForResult(intent, REQUEST_DIRECTORY_ACCESS )
    }

    // Read the contents of the select directory here
    // See this SO question -> https://stackoverflow.com/questions/47941357/how-to-access-files-in-a-directory-given-a-content-uri
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if ( requestCode == REQUEST_DIRECTORY_ACCESS && resultCode == RESULT_OK) {
            val dirUri = data?.data ?: return
            val childrenUri =
                DocumentsContract.buildChildDocumentsUriUsingTree(
                    dirUri,
                    DocumentsContract.getTreeDocumentId( dirUri )
                )
            val tree = DocumentFile.fromTreeUri(this, childrenUri)
            val images = ArrayList<Pair<String,Bitmap>>()
            var errorFound = false
            if ( tree!!.listFiles().isNotEmpty()) {
                for ( doc in tree.listFiles() ) {
                    if ( doc.isDirectory && !errorFound ) {
                        val name = doc.name!!
                        for ( imageDocFile in doc.listFiles() ) {
                            try {
                                images.add( Pair( name , getFixedBitmap( imageDocFile.uri ) ) )
                            }
                            catch ( e : Exception ) {
                                errorFound = true
                                Logger.log( "Could not parse an image in $name directory. Make sure that the file structure is " +
                                        "as described in the README of the project and then restart the app." )
                                break
                            }
                        }
                        Logger.log( "Found ${doc.listFiles().size} images in $name directory" )
                    }
                    else {
                        errorFound = true
                        Logger.log( "The selected folder should contain only directories. Make sure that the file structure is " +
                                "as described in the README of the project and then restart the app." )
                    }
                }
            }
            else {
                errorFound = true
                Logger.log( "The selected folder doesn't contain any directories. Make sure that the file structure is " +
                        "as described in the README of the project and then restart the app." )
            }
            if ( !errorFound ) {
                fileReader.run( images , fileReaderCallback )
                Logger.log( "Detecting faces in ${images.size} images ..." )
            }
            else {
                val alertDialog = AlertDialog.Builder( this ).apply {
                    setTitle( "Error while parsing directory")
                    setMessage( "There were some errors while parsing the directory. Please see the log below. Make sure that the file structure is " +
                            "as described in the README of the project and then tap RESELECT" )
                    setCancelable( false )
                    setPositiveButton( "RESELECT") { dialog, which ->
                        dialog.dismiss()
                        launchChooseDirectoryIntent()
                    }
                    setNegativeButton( "CANCEL" ){ dialog , which ->
                        dialog.dismiss()
                        finish()
                    }
                    create()
                }
                alertDialog.show()
            }

        }
    }

    // Get the image as a Bitmap from given Uri and fix the rotation using the Exif interface
    // Source -> https://stackoverflow.com/questions/14066038/why-does-an-image-captured-using-camera-intent-gets-rotated-on-some-devices-on-a
    private fun getFixedBitmap( imageFileUri : Uri ) : Bitmap {
        var imageBitmap = BitmapUtils.getBitmapFromUri( contentResolver , imageFileUri )
        val exifInterface = ExifInterface( contentResolver.openInputStream( imageFileUri )!! )
        imageBitmap =
            when (exifInterface.getAttributeInt( ExifInterface.TAG_ORIENTATION ,
                ExifInterface.ORIENTATION_UNDEFINED )) {
                ExifInterface.ORIENTATION_ROTATE_90 -> BitmapUtils.rotateBitmap( imageBitmap , 90f )
                ExifInterface.ORIENTATION_ROTATE_180 -> BitmapUtils.rotateBitmap( imageBitmap , 180f )
                ExifInterface.ORIENTATION_ROTATE_270 -> BitmapUtils.rotateBitmap( imageBitmap , 270f )
                else -> imageBitmap
            }
        return imageBitmap
    }

    // ---------------------------------------------- //

    private val fileReaderCallback = object : FileReader.ProcessCallback {
        override fun onProcessCompleted(data: ArrayList<Pair<String, FloatArray>>, numImagesWithNoFaces: Int) {
            frameAnalyser.faceList = data
            saveSerializedImageData( data )
            Logger.log( "Images parsed. Found $numImagesWithNoFaces images with no faces." )
        }
    }

    private fun saveSerializedImageData(data : ArrayList<Pair<String,FloatArray>> ) {
        val serializedDataFile = File( filesDir , SERIALIZED_DATA_FILENAME )
        ObjectOutputStream( FileOutputStream( serializedDataFile )  ).apply {
            writeObject( data )
            flush()
            close()
        }
        sharedPreferences.edit().putBoolean( SHARED_PREF_IS_DATA_STORED_KEY , true ).apply()
    }

    private fun loadSerializedImageData() : ArrayList<Pair<String,FloatArray>> {
        val serializedDataFile = File( filesDir , SERIALIZED_DATA_FILENAME )
        val objectInputStream = ObjectInputStream( FileInputStream( serializedDataFile ) )
        val data = objectInputStream.readObject() as ArrayList<Pair<String,FloatArray>>
        objectInputStream.close()
        return data
    }


}
