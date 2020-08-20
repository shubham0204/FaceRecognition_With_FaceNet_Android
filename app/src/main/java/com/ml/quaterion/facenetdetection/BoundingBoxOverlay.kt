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


    private val displayMetrics = context.resources.displayMetrics

    // Width and height of the device screen in pixels.
    private val dpHeight = displayMetrics.heightPixels
    private val dpWidth = displayMetrics.widthPixels

    // Our boxes will be predicted on a 640 * 480 image. So, we need to scale the boxes to the device screen's width and
    // height
    private val xfactor = dpWidth.toFloat() / 480f
    private val yfactor = dpHeight.toFloat() / 640f
    // Create a Matrix for scaling
    private val output2OverlayTransform = Matrix().apply {
        preScale( xfactor , yfactor )
    }

    // This var is assigned in FrameAnalyer.kt
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

    override fun surfaceChanged(holder: SurfaceHolder?, format: Int, width: Int, height: Int) {
    }

    override fun surfaceDestroyed(holder: SurfaceHolder?) {
    }

    override fun surfaceCreated(holder: SurfaceHolder?) {
    }

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
                Log.e( "Info" , "Rect received ${processedBbox.toShortString()}")
            }
        }
    }

    // Apply the scale transform matrix to the boxes.
    private fun processBBox( bbox : Rect ) : RectF {
        val rectf = RectF( bbox )
        output2OverlayTransform.mapRect( rectf)
        return rectf
    }



}
