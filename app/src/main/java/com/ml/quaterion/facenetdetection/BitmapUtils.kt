/*
 * Copyright 2023 Shubham Panchal
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

import android.content.ContentResolver
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Rect
import android.net.Uri
import android.os.ParcelFileDescriptor
import java.io.File
import java.io.FileDescriptor
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
            val parcelFileDescriptor: ParcelFileDescriptor? = contentResolver.openFileDescriptor(uri, "r")
            val fileDescriptor: FileDescriptor = parcelFileDescriptor!!.fileDescriptor
            val image: Bitmap = BitmapFactory.decodeFileDescriptor(fileDescriptor)
            parcelFileDescriptor.close()
            return image
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


    }

}