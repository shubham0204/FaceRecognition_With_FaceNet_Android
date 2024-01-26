#include <jni.h>
#include <android/log.h>
#include "face_detector.cpp"

extern "C" JNIEXPORT jlong JNICALL
Java_com_ml_shubham0204_facenetdetection_ml_Annotator_createAnnotator(
        JNIEnv* env,
        jobject  ,
        jobjectArray subject_names ,
        jobjectArray subject_embeddings ,
        jint embedding_dim ,
        jfloat threshold_cosine
) {

    // Read subject names
    std::vector<std::string> names ;
    int num_names = env -> GetArrayLength( subject_names ) ;
    jboolean is_copy = JNI_TRUE;
    for( int i = 0 ; i < num_names ; i++ ) {
        jstring j_name = (jstring)( env -> GetObjectArrayElement( subject_names , i ) );
        const char* chars = env -> GetStringUTFChars( j_name , &is_copy ) ;
        std::string name = chars ;
        env -> ReleaseStringUTFChars( j_name , chars ) ;
        names.push_back( name ) ;
    }


    // Read subject_embeddings
    int num_embeddings = env -> GetArrayLength( subject_embeddings ) ;
    if( num_embeddings != num_names ) {
        return -1L ;
    }

    std::vector<float*> embeddings ;
    for( int i = 0 ; i < num_embeddings ; i++ ) {
        jfloatArray embedding = (jfloatArray) env -> GetObjectArrayElement( subject_embeddings , i ) ;
        jfloat* embedding_elements = env -> GetFloatArrayElements( embedding , &is_copy ) ;
        float* embedding_fp = new float[ embedding_dim ] ;
        for( int j = 0 ; j < embedding_dim ; j++ ) {
            *(embedding_fp + j ) = *( embedding_elements + j ) ;
        }
        env -> ReleaseFloatArrayElements( embedding , embedding_elements , 0 ) ;
        env -> DeleteLocalRef( embedding ) ;
        embeddings.push_back( embedding_fp ) ;
    }

    FaceAnnotator* annotator = new FaceAnnotator( names , embeddings , embedding_dim , threshold_cosine ) ;
    return (long) annotator ;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_ml_shubham0204_facenetdetection_ml_Annotator_identify(
        JNIEnv* env,
        jobject  ,
        jlong annotator_ptr ,
        jfloatArray subject_embedding
) {

    FaceAnnotator* annotator = (FaceAnnotator*) annotator_ptr ;
    jboolean is_copy = true;
    jfloat* embedding_elements = env -> GetFloatArrayElements( subject_embedding , &is_copy ) ;
    float* embedding_fp = new float[ annotator -> embedding_dim ] ;
    for( int j = 0 ; j < annotator -> embedding_dim ; j++ ) {
        embedding_fp[ j ] = embedding_elements[ j ] ;
    }
    env -> ReleaseFloatArrayElements( subject_embedding , embedding_elements , 0 ) ;

    std::string label = annotator -> identify( embedding_fp ) ;
    jstring output = env -> NewStringUTF( label.c_str() ) ;

    return output ;
}

extern "C" JNIEXPORT void JNICALL
Java_com_ml_shubham0204_facenetdetection_ml_Annotator_releaseAnnotator(
        JNIEnv* env,
        jobject,
        jlong annotator_ptr) {
    delete ((FaceAnnotator*)annotator_ptr );
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_ml_shubham0204_facenetdetection_ml_FaceNetModel_00024StandardizeOp_standardize(
        JNIEnv* env,
        jobject,
        jfloatArray values ) {

    jboolean is_copy = false;
    uint32_t num_elements = env -> GetArrayLength( values ) ;
    jfloat* elements = env -> GetFloatArrayElements( values , &is_copy ) ;
    float mean = 0.0f;
    for( int i = 0 ; i < num_elements ; i++ ) {
        mean += elements[ i ] ;
    }
    mean = mean / (float) num_elements ;
    float std_dev = 0.0f ;
    for( int i = 0 ; i < num_elements ; i++ ) {
        std_dev += (( elements[ i ] - mean ) * ( elements[ i ] - mean ));
    }
    std_dev = sqrt( std_dev / (float)num_elements ) ;
    for( int i = 0 ; i < num_elements ; i++ ) {
        elements[ i ] = (( elements[i] - mean ) / std_dev) ;
    }
    env -> ReleaseFloatArrayElements( values , elements , 0 ) ;
    return values ;

}