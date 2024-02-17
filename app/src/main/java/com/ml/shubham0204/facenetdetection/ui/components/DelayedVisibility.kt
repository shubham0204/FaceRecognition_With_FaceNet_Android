package com.ml.shubham0204.facenetdetection.ui.components

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.runtime.Composable

@Composable
fun DelayedVisibility( visible: Boolean , content: @Composable (() -> Unit) ) {
    AnimatedVisibility(
        visible = visible ,
        enter = fadeIn(animationSpec = tween(1000)),
        exit = fadeOut(animationSpec = tween(1000))
    ) {
        content()
    }
}