//
//  main.c
//  qwerty
//
//  Created by Parikh Oberoi on 21/01/18.
//  Copyright © 2018 Parikh Oberoi. All rights reserved.
//

#include "csv.h"
#include "matrices.h"
#include <time.h>


typedef struct network_t {
    int ninputs;
    int nhiddens;
    int noutputs;
    int max_steps;
    float learning_rate;
    float *input_to_hidden_weights;
    float *input_to_hidden_delta_weights;
    float *hidden_to_output_weights;
    float *hidden_to_output_back_weights;
    float *hidden_to_output_delta_weights;
    float *inputs;
    float *hiddens;
    float *outputs;
    float *output_errors;
    float *hidden_errors;
    float *hiddens_deltas;
    float *outputs_deltas;
} network_t;

static network_t network;

void init_network( int ninputs, int nhiddens, int noutputs, float learning_rate,
                  int max_steps );
void train_network( int nsamples, int ninputs, int noutputs,
                   float inputs_list[nsamples][ninputs],
                   float targets_list[nsamples][noutputs] );
void query( const float *inputs );


int main() {
    int num_pixels = 28 * 28;
    int ninputs = num_pixels;
    int nhiddens = 100;
    int noutputs = 10;
    float learning_rate = 0.3f;
    int max_steps = 1; 
    int epochs = 10; 
    
    int num_lines = 2000; // more samples = more accurate.
    float labels[num_lines];
    float pixels[num_lines][num_pixels];
    read_csv( "/Users/parikh/Desktop/mnist_train.csv", num_lines, labels, pixels );
#ifdef DRAW_OUT_IMAGES
    for ( int j = 0; j < num_lines; j++ ) 
    { 
        char tmp[128];
        sprintf( tmp, "img%i_label_%i.ppm", j, (int)labels[j] );
        FILE *f = fopen( tmp, "w" );
        assert( f );
        fprintf( f, "P3\n28 28\n255\n" );
        for ( int i = 0; i < num_pixels; i++ ) {
            int v =
            (int)( 255.0f - 255.0f * ( ( pixels[j][i] - 0.01f ) * ( 1.0f / 0.99f ) ) );
            fprintf( f, "%i %i %i\n", v, v, v );
        }
        fclose( f );
    }
#endif
    
    init_network( ninputs, nhiddens, noutputs, learning_rate, max_steps );
    
    for ( int epoch = 0; epoch < epochs; epoch++ ) 
    { 
        float targets_list[num_lines][noutputs];
        for ( int l = 0; l < num_lines; l++ ) {
            for ( int o = 0; o < noutputs; o++ ) {
                if ( (int)labels[l] == o ) {
                    targets_list[l][o] = 1.0f;
                } else {
                    targets_list[l][o] = 0.1f; // lowest useful signal
                }
            }
        }
        
#ifdef DEBUG_PRINT
        printf( "targets list:\n" );
        print_mat( targets_list[5], noutputs, 1 ); // '9'
#endif
        
        train_network( num_lines, ninputs, noutputs, pixels, targets_list );
    }
    

    
    {
        int num_lines = 100;
        float labels[num_lines];
        float pixels[num_lines][num_pixels];
        read_csv( "/Users/parikh/Desktop/mnist_test.csv", num_lines, labels, pixels );
        int sum_correct = 0;
        for ( int i = 0; i < num_lines; i++ ) {
            query( pixels[i] );
            int maxi = 0;
            float maxf = network.outputs[0];
            for ( int j = 1; j < network.noutputs; j++ ) {
                if ( network.outputs[j] > maxf ) {
                    maxf = network.outputs[j];
                    maxi = j;
                }
            }
            printf( "queried. our answer %i (%f conf) - correct answer %i\n", maxi,
                   network.outputs[maxi], (int)labels[i] );
            if ( maxi == (int)labels[i] ) {
                sum_correct++;
            }
#ifdef DEBUG_PRINT
            print_mat( network.outputs, network.noutputs, 1 );
#endif
          
        }
        float accuracy = (float)sum_correct / (float)num_lines;
        printf( "total accuracy = %f\n", accuracy );
    }
    return 0;
}

void init_network( int ninputs, int nhiddens, int noutputs, float learning_rate,
                  int max_steps ) {
    printf( "initialising...\n" );

    
    network.ninputs = ninputs;
    network.nhiddens = nhiddens;
    network.noutputs = noutputs;
    network.max_steps = max_steps;
    network.learning_rate = learning_rate;
    
    {
        size_t sz_a = sizeof( float ) * ninputs * nhiddens;
        size_t sz_b = sizeof( float ) * nhiddens * noutputs;
        size_t sz_inputs = sizeof( float ) * ninputs;
        size_t sz_hiddens = sizeof( float ) * nhiddens;
        size_t sz_outputs = sizeof( float ) * noutputs;
      
        network.input_to_hidden_weights = (float *)malloc( sz_a );
        assert( network.input_to_hidden_weights );
        network.input_to_hidden_delta_weights = (float *)malloc( sz_a );
        assert( network.input_to_hidden_delta_weights );
        network.hidden_to_output_weights = (float *)malloc( sz_b );
        assert( network.hidden_to_output_weights );
        network.hidden_to_output_back_weights = (float *)malloc( sz_b );
        assert( network.hidden_to_output_back_weights );
        network.hidden_to_output_delta_weights = (float *)malloc( sz_b );
        assert( network.hidden_to_output_delta_weights );
        // vectors for each layer
        network.inputs = (float *)calloc( ninputs, sizeof( float ) );
        assert( network.inputs );
        network.hiddens = (float *)calloc( nhiddens, sizeof( float ) );
        assert( network.hiddens );
        network.hidden_errors = (float *)calloc( nhiddens, sizeof( float ) );
        assert( network.hidden_errors );
        network.outputs = (float *)calloc( noutputs, sizeof( float ) );
        assert( network.outputs );
        network.output_errors = (float *)calloc( noutputs, sizeof( float ) );
        assert( network.output_errors );
        network.hiddens_deltas = (float *)calloc( nhiddens, sizeof( float ) );
        network.outputs_deltas = (float *)calloc( noutputs, sizeof( float ) );
        size_t sz_total =
        sz_a * 2 + sz_b * 3 + sz_inputs + sz_hiddens * 3 + sz_outputs * 3;
        printf( "allocated %lu bytes (%lu kB) (%lu MB)\n", sz_total, sz_total / 1024,
               sz_total / ( 1024 * 1024 ) );
#ifdef DEBUG_PRINT
        printf( "  inputs %lu\n", sz_inputs );
        printf( "  hiddens %lu\n", sz_hiddens );
        printf( "  hidden errors %lu\n", sz_hiddens );
        printf( "  hidden deltas %lu\n", sz_hiddens );
        printf( "  outputs %lu\n", sz_outputs );
        printf( "  output errors %lu\n", sz_outputs );
        printf( "  output deltas %lu\n", sz_outputs );
        printf( "  weights input->hidden %lu\n", sz_a );
        printf( "  delta weights input->hidden %lu\n", sz_a );
        printf( "  weights hidden->outputs %lu\n", sz_b );
        printf( "  weights outputs->hidden %lu\n", sz_b );
        printf( "  delta weights hidden->outputs %lu\n", sz_b );
#endif
    }
    
    randomise_mat( network.input_to_hidden_weights, ninputs, nhiddens );
    randomise_mat( network.hidden_to_output_weights, nhiddens, noutputs );
}

void train_network( int nsamples, int ninputs, int noutputs,
                   float inputs_list[nsamples][ninputs],
                   float targets_list[nsamples][noutputs] ) {
    assert( inputs_list );
    assert( targets_list );
    
    printf( "training...\n" );
    
    
    // samples loop here
    for ( int curr_sample = 0; curr_sample < nsamples; curr_sample++ ) {
        
       
        for ( int step = 0; step < network.max_steps; step++ ) {
            
            { 
                memcpy( network.inputs, inputs_list[curr_sample],
                       network.ninputs * sizeof( float ) );
              
                mult_mat_vec( network.input_to_hidden_weights, network.nhiddens,
                             network.ninputs, network.inputs, network.hiddens );
                sigmoid( network.hiddens, network.hiddens, network.nhiddens );
                
                mult_mat_vec( network.hidden_to_output_weights, network.noutputs,
                             network.nhiddens, network.hiddens, network.outputs );
                sigmoid( network.outputs, network.outputs, network.noutputs );
            }
            { 
                for ( int i = 0; i < network.noutputs; i++ ) {
                    network.output_errors[i] =
                    targets_list[curr_sample][i] - network.outputs[i];
                    
                    // printf( "errors[%i] = %f\n", i, network.output_errors[i] );
                }
                
                transpose_mat( network.hidden_to_output_weights,
                              network.hidden_to_output_back_weights, network.nhiddens,
                              network.noutputs );
                
                mult_mat_vec( network.hidden_to_output_back_weights, network.nhiddens,
                             network.noutputs, network.output_errors,
                             network.hidden_errors );
                
                { // adjust hidden->output weights
                    for ( int i = 0; i < network.noutputs; i++ ) {
                        network.outputs_deltas[i] = network.output_errors[i] *
                        network.outputs[i] *
                        ( 1.0f - network.outputs[i] );
                        //    printf( "output delta[%i] = %f\n", i, network.outputs_deltas[i] );
                    }
                    colrow_vec_mult( network.outputs_deltas, network.hiddens,
                                    network.noutputs, network.nhiddens,
                                    network.hidden_to_output_delta_weights );
                    
                    //    printf( "delta weights matrix:\n" );
                    //    print_mat( network.hidden_to_output_delta_weights, network.noutputs,
                    //                         network.nhiddens );
                    
                    for ( int i = 0; i < ( network.nhiddens * network.noutputs ); i++ ) {
                        network.hidden_to_output_delta_weights[i] *= network.learning_rate;
                        //    printf( "output weight %i before = %f\n", i,
                        //                    network.hidden_to_output_weights[i] );
                        network.hidden_to_output_weights[i] +=
                        network.hidden_to_output_delta_weights[i];
                        //    printf( "output weight %i after = %f\n", i,
                        //                    network.hidden_to_output_weights[i] );
                    }
                }
                
                { // adjust input->hidden weights
                    for ( int i = 0; i < network.nhiddens; i++ ) {
                        network.hiddens_deltas[i] = network.hidden_errors[i] *
                        network.hiddens[i] *
                        ( 1.0f - network.hiddens[i] );
                        //    printf( "hidden delta[%i] = %f\n", i, network.hiddens_deltas[i] );
                    }
                    colrow_vec_mult( network.hiddens_deltas, network.inputs, network.nhiddens,
                                    network.ninputs, network.input_to_hidden_delta_weights );
                    
                    // printf( "delta weights matrix:\n" );
                    //    print_mat( network.input_to_hidden_delta_weights, network.nhiddens,
                    //                     network.ninputs );
                    
                    for ( int i = 0; i < ( network.ninputs * network.nhiddens ); i++ ) {
                        network.input_to_hidden_delta_weights[i] *= network.learning_rate;
                        
                        //    printf( "output weight %i before = %f\n", i,
                        //                network.input_to_hidden_weights[i] );
                        network.input_to_hidden_weights[i] +=
                        network.input_to_hidden_delta_weights[i];
                        
                        //    printf( "output weight %i after = %f\n", i,
                        //                network.input_to_hidden_weights[i] );
                    }
                }
                
            } // end back-propagation
            
#ifdef PRINT_TRAINING
            if ( step % 100 == 0 ) {
                printf( "end of step %i\n", step );
                float error_sum = 0.0f;
                for ( int i = 0; i < network.noutputs; i++ ) {
                    printf( "output[%i] = %f target = %f\n", i, network.outputs[i],
                           targets_list[curr_sample][i] );
                    error_sum += ( targets_list[curr_sample][i] - network.outputs[i] );
                }
                printf( "error sum was %f\n", error_sum );
            }
#endif
        }
    } 
    
   
}


void query( const float *inputs ) {
    assert( inputs );
    
    printf( "querying...\n" );
    memcpy( network.inputs, inputs, network.ninputs * sizeof( float ) );
    memset( network.outputs, 0, network.noutputs );

    mult_mat_vec( network.input_to_hidden_weights, network.nhiddens, network.ninputs,
                 network.inputs, network.hiddens );
    
    sigmoid( network.hiddens, network.hiddens, network.nhiddens );
  
    mult_mat_vec( network.hidden_to_output_weights, network.noutputs,
                 network.nhiddens, network.hiddens, network.outputs );

    sigmoid( network.outputs, network.outputs, network.noutputs );
}


