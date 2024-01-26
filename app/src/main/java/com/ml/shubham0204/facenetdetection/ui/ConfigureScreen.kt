package com.ml.shubham0204.facenetdetection.ui

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
import androidx.compose.material3.Checkbox
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
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.ml.shubham0204.facenetdetection.ml.ModelInfo
import com.ml.shubham0204.facenetdetection.ml.Models

private val userPreferences = object {
    var faceDetectionModel: MutableState<ModelInfo> = mutableStateOf( Models.FACENET )
    var isWritingLogs: MutableState<Boolean> = mutableStateOf( false )
    var nnApiEnabled: MutableState<Boolean> = mutableStateOf( true )
    var gpuEnabled: MutableState<Boolean> = mutableStateOf( true )
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
    }
}

@Composable
private fun Preference(
    title: String ,
    description: String = "" ,
    content: @Composable ( ColumnScope.() -> Unit)
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
private fun PreferenceDelegates() {
    var nnApiEnabled by remember{ userPreferences.nnApiEnabled }
    var gpuEnabled by remember{ userPreferences.gpuEnabled }
    Preference(
        title = "TensorFlow Lite Delegates" ,
        description = "NNAPIDelegate and GPUDelegate"
    ) {
        Checkbox(
            checked = nnApiEnabled,
            onCheckedChange = {
                nnApiEnabled = it
            }
        )
        Column {
            Text(text = "NN API")
            Text(text = "Logging writes log in a text file" )
        }
        Checkbox(
            checked = gpuEnabled,
            onCheckedChange = {
                gpuEnabled = it
            }
        )
        Column {
            Text(text = "GPU Delegate")
            Text(text = "Logging writes log in a text file" )
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

