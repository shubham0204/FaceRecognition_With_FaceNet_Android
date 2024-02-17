package com.ml.shubham0204.facenetdetection

import android.content.ContentResolver
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Rect
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.io.FileOutputStream

// Helper class for operations on Bitmaps
class BitmapUtils {

    companion object {

        // Crop the given bitmap with the given rect.
        fun cropRectFromBitmap(
            source: Bitmap,
            rect: Rect
        ): Bitmap {
            return Bitmap.createBitmap( source , rect.left , rect.top , rect.width() , rect.height() )
        }

        fun validateRect(
            cameraFrameBitmap: Bitmap ,
            boundingBox: Rect
        ) : Boolean {
            return boundingBox.left >= 0 &&
                    boundingBox.top >= 0 &&
                    (boundingBox.left + boundingBox.width()) < cameraFrameBitmap.width &&
                    (boundingBox.top + boundingBox.height()) < cameraFrameBitmap.height
        }


        // Get the image as a Bitmap from given Uri
        // Source -> https://developer.android.com/training/data-storage/shared/documents-files#bitmap
        fun getBitmapFromUri(
            contentResolver : ContentResolver ,
            uri: Uri
        ): Bitmap {
            return BitmapFactory.decodeStream( contentResolver.openInputStream( uri ) )
        }


        // Rotate the given `source` by `degrees`.
        // See this SO answer -> https://stackoverflow.com/a/16219591/10878733
        fun rotateBitmap(
            source: Bitmap ,
            degrees : Float
        ): Bitmap {
            val matrix = Matrix()
            matrix.postRotate( degrees )
            return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix , false )
        }


        // Use this method to save a Bitmap to the internal storage ( app-specific storage ) of your device.
        // To see the image, go to "Device File Explorer" -> "data" -> "data" -> "com.ml.quaterion.facenetdetection" -> "files"
        fun saveBitmap(
            context: Context,
            image: Bitmap,
            name: String
        ) {
            val fileOutputStream = FileOutputStream(File( context.filesDir.absolutePath + "/$name.png"))
            image.compress(Bitmap.CompressFormat.PNG, 100, fileOutputStream)
        }


        // Get the image as a Bitmap from given Uri and fix the rotation using the Exif interface
        // Source -> https://stackoverflow.com/questions/14066038/why-does-an-image-captured-using-camera-intent-gets-rotated-on-some-devices-on-a
        fun getFixedBitmap(
            context: Context ,
            imageFileUri : Uri ) : Bitmap {
            var imageBitmap = getBitmapFromUri( context.contentResolver , imageFileUri )
            val exifInterface = ExifInterface( context.contentResolver.openInputStream( imageFileUri )!! )
            imageBitmap =
                when (exifInterface.getAttributeInt( ExifInterface.TAG_ORIENTATION ,
                    ExifInterface.ORIENTATION_UNDEFINED )) {
                    ExifInterface.ORIENTATION_ROTATE_90 -> rotateBitmap( imageBitmap , 90f )
                    ExifInterface.ORIENTATION_ROTATE_180 -> rotateBitmap( imageBitmap , 180f )
                    ExifInterface.ORIENTATION_ROTATE_270 -> rotateBitmap( imageBitmap , 270f )
                    else -> imageBitmap
                }
            return imageBitmap
        }

    }

}