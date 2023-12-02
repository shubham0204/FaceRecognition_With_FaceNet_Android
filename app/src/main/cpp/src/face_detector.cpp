#include <jni.h>
#include <string>
#include <vector>
#include <math.h>
#include <unordered_map>
#include <algorithm>
#include <android/log.h>

class FaceAnnotator {

    float cosine_similarity(
        uint32_t embedding_dim ,
        const float embedding1[] ,
        const float embedding2[]
    ) {
        float mag1 = 0.0f ;
        float mag2 = 0.0f ;
        float prod = 0.0f ;
        for( int i = 0 ; i < embedding_dim ; i++ ) {
            prod += (embedding1[i] * embedding2[i]) ;
            mag1 += (embedding1[i] * embedding1[i]) ;
            mag2 += (embedding2[i] * embedding2[i]) ;
        }
        return prod / (sqrt( mag1 ) * sqrt( mag2 )) ;
    }

    float cosine_similarity_v2(
        uint32_t embedding_dim ,
        const float embedding1[] ,
        const float embedding2[] ,
        float mag2
    ) {
        float mag1 = 0.0f ;
        float prod = 0.0f ;
        for( int i = 0 ; i < embedding_dim ; i++ ) {
            prod += (embedding1[i] * embedding2[i]) ;
            mag1 += (embedding1[i] * embedding1[i]) ;
        }
        return prod / ( sqrt( mag1 ) * mag2 ) ;
    }

    float l2_norm(
        uint32_t embedding_dim ,
        const float embedding1[] ,
        const float embedding2[]
    ) {
        float mag1 = 0.0f ;
        float mag2 = 0.0f ;
        float prod = 0.0f ;
        for( int i = 0 ; i < embedding_dim ; i++ ) {
            prod += (embedding1[i] * embedding2[i]) ;
            mag1 += (embedding1[i] * embedding1[i]) ;
            mag2 += (embedding2[i] * embedding2[i]) ;
        }
        return prod / (sqrt( mag1 ) * sqrt( mag2 )) ;
    }

    float magnitude(
       uint32_t embedding_dim ,
       const float embedding[]
    ) {
        float mag = 0.0f ;
        for( int i = 0 ; i < embedding_dim ; i++ ) {
            mag += (embedding[i] * embedding[i]) ;
        }
        return sqrt( mag ) ;
    }

    public:

    std::vector<std::string> subject_names ;
    std::vector<float*> subject_embeddings ;
    std::vector<float> subject_embedding_mags ;
    std::unordered_map<std::string,float> name_score_map ;
    std::unordered_map<std::string,uint32_t> name_freq_map ;
    uint32_t embedding_dim ;
    float threshold_cosine_similarity ;
    float threshold_l2_norm ;
    float (FaceAnnotator::*score_func_ptr)( uint32_t , const float[] , const float[] ) ;

    FaceAnnotator(
        std::vector<std::string> subject_names ,
        std::vector<float*> embeddings ,
        uint32_t embedding_dim ,
        float threshold_cosine_similarity ,
        float threshold_l2_norm ,
        std::string method
    ) {
        this -> subject_names = subject_names ;
        this -> subject_embeddings = embeddings ;
        this -> embedding_dim = embedding_dim ;
        this -> threshold_cosine_similarity = threshold_cosine_similarity ;
        this -> threshold_l2_norm = threshold_l2_norm ;
        for( const auto embedding : subject_embeddings ) {
            this -> subject_embedding_mags.push_back( this -> magnitude( embedding_dim , embedding ) ) ;
        }
        for( const auto & subject_name : subject_names ) {
            if( this -> name_freq_map.find( subject_name ) == this -> name_freq_map.end() ) {
                this -> name_freq_map[ subject_name ] = 1 ;
            }
            else {
                this -> name_freq_map[ subject_name ] += 1 ;
            }
        }
        for( auto const& x: this -> name_freq_map ) {
            this -> name_score_map[ x.first ] = 0.0f ;
        }
        if( method == "cosine" ) {
            this -> score_func_ptr = &FaceAnnotator::cosine_similarity ;
            // score_cmp_func_ptr = [=]( const float& a , const float& b ) { return a < b ; } ;
        }
        else {
            this -> score_func_ptr = &FaceAnnotator::l2_norm ;
        }
    }

    std::string identify(
        float* candidate_embedding
    ) {
        for( int i = 0 ; i < subject_names.size() ; i++ ) {
            //float score = (this->*score_func_ptr)( this -> embedding_dim ,
            //                                       candidate_embedding ,
            //                                      this -> subject_embeddings[i] ) ;
            float score = this ->cosine_similarity_v2(
                                this -> embedding_dim ,
                                candidate_embedding ,
                                this -> subject_embeddings[i] ,
                                this -> subject_embedding_mags[i]
                                ) ;
            //__android_log_write(ANDROID_LOG_ERROR, "Error from native", subject_names[i].c_str() );
            //__android_log_write(ANDROID_LOG_ERROR, "Error from native", std::to_string( score ).c_str() );
            name_score_map[ subject_names[i] ] += score ;
        }
        std::string max_score_name = "UNKNOWN" ;
        float max_score = -(1e+3) ;
        for( auto x: name_score_map ) {
            float score = x.second / (float) name_freq_map[ x.first ] ;
            if( score > max_score ) {
                max_score = score ;
                max_score_name = x.first ;
            }
            x.second = 0.0f ;
        }
        if( max_score >= this -> threshold_cosine_similarity ) {
            return max_score_name ;
        }
        else {
            return "UNKNOWN" ;
        }
    }

} ;

extern "C" JNIEXPORT jlong JNICALL
Java_com_ml_quaterion_facenetdetection_ml_Annotator_createAnnotator(
        JNIEnv* env,
        jobject  ,
        jobjectArray subject_names ,
        jobjectArray subject_embeddings ,
        jint embedding_dim ,
        jfloat threshold_cosine ,
        jfloat threshold_l2 ,
        jstring _method
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

    // Read method
    const char* method_chars = env -> GetStringUTFChars( _method , &is_copy ) ;
    std::string method = method_chars ;
    env -> ReleaseStringUTFChars( _method , method_chars ) ;

    FaceAnnotator* annotator = new FaceAnnotator( names , embeddings , embedding_dim , threshold_cosine , threshold_l2 , method ) ;
    return (long) annotator ;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_ml_quaterion_facenetdetection_ml_Annotator_identify(
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
Java_com_ml_quaterion_facenetdetection_ml_Annotator_releaseAnnotator(
        JNIEnv* env,
        jobject,
        jlong annotator_ptr) {
    delete ((FaceAnnotator*)annotator_ptr );
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_ml_quaterion_facenetdetection_ml_FaceNetModel_00024StandardizeOp_standardize(
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

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_ml_quaterion_facenetdetection_ml_FaceNetModel_standardize(
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

