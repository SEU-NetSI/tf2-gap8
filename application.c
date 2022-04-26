#include "application.h"
#include "mnistKernels.h"
#include "gaplib/ImgIO.h"
#include "pmsis.h"
#include "stdio.h"


#define IMG_W 224
#define IMG_H 224
#define IMG_C 3
#define NUM_CLASSES 2
#define __XSTR(__s) __STR(__s)
#define __STR(__s) #__s

static unsigned char * Input_1;
static signed short * Output_1;
static struct pi_device cluster_dev;
static struct pi_cluster_conf conf;
static struct pi_cluster_task *task;

AT_HYPERFLASH_FS_EXT_ADDR_TYPE __PREFIX(_L3_Flash) = 0;


static void cluster()
{
    __PREFIX(CNN)(Input_1, Output_1);
    printf("Model Run completed\n");


    //resolve output
    int res = 0;
    signed short max = Output_1[0];
    for(int i = 0;i<NUM_CLASSES;i++)
    {
        printf("class %d: %ld\n", i, Output_1[i]);
        if(Output_1[i] > max)
        {
            max = Output_1[i];
            res = i;
        }
    }
    printf("\n");
    printf("result of model: %d\n", res);
}


int application()
{

    printf("\n\t *** NNTOOL application *** \n\n");

    Input_1 = (unsigned char *)pi_l2_malloc((IMG_H*IMG_W*IMG_C)*sizeof(unsigned char));
    if(Input_1==NULL)
    {
        printf("Fail to allocate memory for image\n");
        pmsis_exit(-1);
    }
    printf("Allocate memory for image\n");

    Output_1 = (signed short *)pi_l2_malloc(NUM_CLASSES*sizeof(signed short));
    if(Output_1==NULL)
    {
        printf("Fail to allocate memory for Output_1\n");
        pmsis_exit(-2);
    }
    printf("Allocate memory for Output_1\n");

    


    // read image from file
    char *ImageName = __XSTR(AT_IMAGE);
    if (ReadImageFromFile(ImageName, IMG_W, IMG_H, IMG_C, Input_1, sizeof(char)*(IMG_W*IMG_H*IMG_C), IMGIO_OUTPUT_CHAR, 0))
    {
        printf("Failed to load image %s\n", ImageName);
        pmsis_exit(-3);
    }
    
    /*Configure Cluster Task*/

    pi_cluster_conf_init(&conf);
    pi_open_from_conf(&cluster_dev, (void*)&conf);

    
    if (pi_cluster_open(&cluster_dev))
    {
        printf("Cluster open failed !\n");
        pmsis_exit(-4);
    }
    printf("Cluster open\n");

    
    task = pi_l2_malloc(sizeof(struct pi_cluster_task));
    if(task==NULL)
    {
        printf("Fail to allocate memory for cluster task !\n");
        pmsis_exit(-5);
    }
    printf("Allocate memory for cluster task\n");
    
    
    memset(task, 0, sizeof(struct pi_cluster_task));
    task->entry = &cluster;
    task->stack_size = STACK_SIZE;             // defined in makefile
    task->slave_stack_size = SLAVE_STACK_SIZE; // "
    task->arg = NULL;

    
    printf("Constructor\n");
    // IMPORTANT - MUST BE CALLED AFTER THE CLUSTER IS SWITCHED ON!!!!
    if(__PREFIX(CNN_Construct)())
    {
        printf("Graph constructor exited with an error\n");
        pmsis_exit(-5);
    }
    
    printf("Call cluster\n");
    pi_cluster_send_task_to_cl(&cluster_dev, task);

    __PREFIX(CNN_Destruct)();




    // Close the cluster
    pi_cluster_close(&cluster_dev);
    pi_l2_free(task, (sizeof(struct pi_cluster_task)));
    pi_l2_free(Output_1, (NUM_CLASSES*sizeof(signed short)));
    pi_l2_free(Input_1, ((IMG_H*IMG_W)*sizeof(unsigned char)));

    pmsis_exit(0);

    return 0;
}



int main(void)
{
    return pmsis_kickoff((void *)application);
}