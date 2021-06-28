/*
 * Copyright 2020 Shubham Panchal
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

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.util.Log
import android.view.SurfaceHolder
import android.view.SurfaceView

// Defines an overlay on which the boxes and text will be drawn.
class BoundingBoxOverlay( context: Context , attributeSet: AttributeSet )
    : SurfaceView( context , attributeSet ) , SurfaceHolder.Callback {

    // DisplayMetrics for the current display
    private val displayMetrics = context.resources.displayMetrics

    // Width and height of the device screen in pixels.
    private val dpHeight = displayMetrics.heightPixels
    private val dpWidth = displayMetrics.widthPixels

    // Our boxes will be predicted on a 640 * 480 image. So, we need to scale the boxes to the device screen's width and
    // height
    private val xfactor = dpWidth.toFloat() / 480f
    private val yfactor = dpHeight.toFloat() / 640f

    // Create a Matrix for scaling the bbox coordinates ( for REAR camera )
    private val output2OverlayTransformRearLens = Matrix().apply {
        preScale( xfactor , yfactor )
    }

    override fun surfaceCreated(holder: SurfaceHolder) {
        TODO("Not yet implemented")
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        TODO("Not yet implemented")
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        TODO("Not yet implemented")
    }

    // Create a Matrix for scaling the bbox coordinates ( for FRONT camera )
    // For the front camera, we need to have an additional postScale(), so as to avoid
    // mirror images of boxes.
    private val output2OverlayTransformFrontLens = Matrix().apply {
        preScale( xfactor , yfactor )
        postScale( -1f , 1f , dpWidth/2f , dpHeight/2f )
    }

    // This var is assigned in FrameAnalyser.kt
    var faceBoundingBoxes : ArrayList<Prediction>? = null

    // Defines a Paint object for the boxes.
    private val boxPaint = Paint().apply {
        color = Color.parseColor( "#4D90caf9" )
        style = Paint.Style.FILL
    }
    // Defines a Paint object for the text.
    private val textPaint = Paint().apply {
        strokeWidth = 2.0f
        textSize = 32f
        color = Color.WHITE
    }

    // Determines which Matrix should be used for transformation.
    // See MainActivity.kt for its uses.
    var addPostScaleTransform = false


    override fun onDraw(canvas: Canvas?) {
        if ( faceBoundingBoxes != null ) {
            for ( face in faceBoundingBoxes!!) {
                val processedBbox = processBBox( face.bbox )
                // Draw boxes and text
                canvas?.drawRoundRect( processedBbox , 16f , 16f , boxPaint )
                canvas?.drawText(
                    face.label ,
                    processedBbox.centerX() ,
                    processedBbox.centerY() ,
                    textPaint
                )
            }
        }
    }

    // Apply the scale transform matrix to the boxes.
    private fun processBBox( bbox : Rect ) : RectF {
        val rectf = RectF( bbox )
        // Add suitable Matrix transform
        when ( addPostScaleTransform ) {
            true -> output2OverlayTransformFrontLens.mapRect( rectf )
            false -> output2OverlayTransformRearLens.mapRect( rectf )
        }
        return rectf
    }

}
