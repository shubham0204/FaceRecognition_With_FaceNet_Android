package com.ml.quaterion.facenetdetection.ui

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.Orientation
import androidx.compose.foundation.gestures.scrollable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.ml.quaterion.facenetdetection.ml.ModelInfo
import com.ml.quaterion.facenetdetection.ml.Models

private val userPreferences = object {
    var faceDetectionModel: MutableState<ModelInfo> = mutableStateOf( Models.FACENET )
    var distanceMetric: MutableState<Models.Companion.DistanceMetrics> = mutableStateOf( Models.Companion.DistanceMetrics.COSINE )
    var isWritingLogs: MutableState<Boolean> = mutableStateOf( false )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ConfigureScreen(
    onNavigateToMainScreen: (() -> Unit)
) {
    Surface( modifier = Modifier
        .background(Color.White)
        .fillMaxSize() ) {
        Scaffold(
            topBar = {
                TopAppBar(
                    colors = TopAppBarDefaults.topAppBarColors(
                        containerColor = MaterialTheme.colorScheme.primaryContainer,
                        titleContentColor = MaterialTheme.colorScheme.primary,
                    ),
                    title = { Text(text = "App Bar Title") } ,
                    navigationIcon = {
                        IconButton(onClick = { onNavigateToMainScreen() }) {
                            Icon(
                                imageVector = Icons.Default.ArrowBack,
                                contentDescription = "Navigate back"
                            )
                        }
                    }
                )
            }
        ) {
            ScreenUI( it )
        }
    }
}

@Composable
private fun ScreenUI( paddingValues: PaddingValues ) {
    Column(
        Modifier
            .padding(
                start = 16.dp,
                end = 16.dp,
                top = paddingValues.calculateTopPadding(),
                bottom = 16.dp
            )
            .scrollable(rememberScrollState(), Orientation.Vertical)
            .fillMaxWidth()) {
        PreferenceChooseModel()
        PreferenceWriteLogs()
        PreferenceChooseImagesDir()
        PreferenceChooseDistanceMetric()
    }
}

@Composable
private fun Preference(
    title: String ,
    description: String = "" ,
    content: @Composable() ( ColumnScope.() -> Unit)
) {
    Card(
        modifier = Modifier.fillMaxWidth() ,
        border = BorderStroke( 1.dp , Color.LightGray ) ,
        colors = CardDefaults.cardColors( containerColor = Color.White)
    ) {
        Text( modifier = Modifier.padding( 16.dp ) ,  text = title )
        if( description != "" ) {
            Text( modifier = Modifier.padding( 16.dp ) , text = description )
        }
        content( this )
    }
}

@Composable
private fun PreferenceChooseModel() {
    val selectedPreference by remember{ userPreferences.faceDetectionModel }
    Preference(
        title = "Face Detection Model" ,
        description = "Some description will go here"
    ) {
        Models.models.forEach {
            Row(
                verticalAlignment = Alignment.CenterVertically ,
                modifier = Modifier
                    .padding(16.dp)
                    .fillMaxWidth()
                    .clickable {
                        userPreferences.faceDetectionModel.value = it
                    },
            ){
                RadioButton(selected = it.name == selectedPreference.name , onClick = {} )
                Column {
                    Text(text = it.name)
                    Text(text = it.description)
                }
            }
        }
    }
}

@Composable
private fun PreferenceWriteLogs() {
    val writeLogs by remember{ userPreferences.isWritingLogs }
    Preference(
        title = "Write logs" ,
        description = "Write attendance logs"
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically ,
            modifier = Modifier
                .padding(16.dp)
                .fillMaxWidth()
                .clickable {
                    userPreferences.isWritingLogs.value = userPreferences.isWritingLogs.value.not()
                },
        ){
            RadioButton(
                selected = writeLogs ,
                onClick = {} )
            Column {
                Text(text = "Enable logs")
                Text(text = "Logging writes log in a text file" )
            }
        }
    }
}


@Composable
private fun PreferenceChooseImagesDir() {
    Preference(
        title = "Choose images directory" ,
        description = "A directory which contains images"
    ) {

    }
}

@Composable
private fun PreferenceChooseDistanceMetric() {
    val selectedPreference by remember{ userPreferences.distanceMetric }
    Preference(
        title = "Distance Metric" ,
        description = "L2 Norm or Cosine Similarity"
    ) {
        Models.models.forEach {
            Row(
                verticalAlignment = Alignment.CenterVertically ,
                modifier = Modifier
                    .padding(16.dp)
                    .fillMaxWidth()
                    .clickable {
                        userPreferences.faceDetectionModel.value = it
                    },
            ){
                RadioButton(selected = it.name == selectedPreference.name , onClick = { /*TODO*/ })
                Column {
                    Text(text = it.name)
                    Text(text = it.description)
                }
            }
        }
    }
}

