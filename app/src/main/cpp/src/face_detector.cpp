#include <string>
#include <vector>
#include <math.h>
#include <unordered_map>
#include <algorithm>

class FaceAnnotator {

    float cosine_similarity(
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

    FaceAnnotator(
        std::vector<std::string> subject_names ,
        std::vector<float*> embeddings ,
        uint32_t embedding_dim ,
        float threshold_cosine_similarity
    ) {
        this -> subject_names = subject_names ;
        this -> subject_embeddings = embeddings ;
        this -> embedding_dim = embedding_dim ;
        this -> threshold_cosine_similarity = threshold_cosine_similarity ;
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
    }

    std::string identify(
        float* candidate_embedding
    ) {
        for( int i = 0 ; i < subject_names.size() ; i++ ) {
            float score = this -> cosine_similarity(
                    this -> embedding_dim ,
                    candidate_embedding ,
                    this -> subject_embeddings[i] ,
                    this -> subject_embedding_mags[i]
                    ) ;
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


