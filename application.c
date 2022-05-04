#include "application.h"
#include "modelKernels.h"
#include "gaplib/ImgIO.h"

#include "bsp/camera/himax.h"

#include "pmsis.h"
#include "stdio.h"

//#define USE_CAMERA

#define CAMERA_WIDTH 324
#define CAMERA_HEIGHT 244

#define IMG_W 224
#define IMG_H 224
#define IMG_C 3
#define NUM_CLASSES 2
#define __XSTR(__s) __STR(__s)
#define __STR(__s) #__s

static unsigned char * Input_1;
static signed short * Output_1;

static struct pi_device cam;
static struct pi_device cluster_dev;
static struct pi_cluster_conf conf;

static pi_task_t cam_task;
static struct pi_cluster_task *task;

AT_HYPERFLASH_FS_EXT_ADDR_TYPE __PREFIX(_L3_Flash) = 0;


static void cluster()
{
    __PREFIX(CNN)(Input_1, Output_1);
    printf("Model Run completed\n");
}


static int32_t open_camera_himax(struct pi_device *device)
{
    struct pi_himax_conf cam_conf;
    //初始化himax相机配置
    pi_himax_conf_init(&cam_conf);

    cam_conf.format = PI_CAMERA_QVGA;

    pi_open_from_conf(device, &cam_conf);
    if (pi_camera_open(device))
    {
        return -1;
    }


    // Rotate camera orientation
    pi_camera_control(device, PI_CAMERA_CMD_START, 0);
    uint8_t set_value = 3;
    uint8_t reg_value;

    pi_camera_reg_set(device, IMG_ORIENTATION, &set_value);
    pi_time_wait_us(1000000);
    pi_camera_reg_get(device, IMG_ORIENTATION, &reg_value);
    if (set_value!=reg_value)
    {
        printf("Failed to rotate camera image\n");
        return -1;
    }
    pi_camera_control(device, PI_CAMERA_CMD_STOP, 0);

    pi_camera_control(device, PI_CAMERA_CMD_AEG_INIT, 0);

    return 0;
}


static void resolve_output()
{
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


static void cam_handler(void *arg)
{
    //采集完一张图片停止捕获
    pi_camera_control(&cam, PI_CAMERA_CMD_STOP, 0);

    printf("Call cluster\n");
    pi_cluster_send_task_to_cl(&cluster_dev, task);
    resolve_output();
    
    pi_task_callback(&cam_task, cam_handler, NULL);
    pi_camera_capture_async(&cam, Input_1, CAMERA_WIDTH * CAMERA_HEIGHT, &cam_task);
    pi_camera_control(&cam, PI_CAMERA_CMD_START, 0);
}


int application()
{
    printf("\n\t *** NNTOOL application *** \n\n");
    int errors = 0;
#ifdef USE_CAMERA
    Input_1 = (unsigned char *)pi_l2_malloc((CAMERA_HEIGHT*CAMERA_WIDTH)*sizeof(unsigned char));
    if(Input_1==NULL)
    {
        printf("Fail to allocate memory for Camera Buffer\n");        
        pmsis_exit(-1);
    }
    printf("Allocate memory for Camera\n");
#else
    Input_1 = (unsigned char *)pi_l2_malloc((IMG_H*IMG_W*IMG_C)*sizeof(unsigned char));
    if(Input_1==NULL)
    {
        printf("Fail to allocate memory for image\n");        
        pmsis_exit(-1);
    }
    printf("Allocate memory for image\n");
#endif

    Output_1 = (signed short *)pi_l2_malloc(NUM_CLASSES*sizeof(signed short));
    if(Output_1==NULL)
    {
        printf("Fail to allocate memory for Output_1\n");
        pmsis_exit(-2);
    }
    printf("Allocate memory for Output_1\n");


#ifdef USE_CAMERA    
    errors = open_camera_himax(&cam);
    if (errors)
    {
        printf("Failed to open camera : %ld\n", errors);
        pmsis_exit(-2);
    }
#else
    // read image from file
    char *ImageName = __XSTR(AT_IMAGE);
    if (ReadImageFromFile(ImageName, IMG_W, IMG_H, IMG_C, Input_1, sizeof(char)*(IMG_W*IMG_H*IMG_C), IMGIO_OUTPUT_CHAR, 0))
    {
        printf("Failed to load image %s\n", ImageName);
        pmsis_exit(-3);
    }
#endif


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
    errors = __PREFIX(CNN_Construct)();
    printf("%d\n", errors); // error id =3 L2 is used up
    if(errors)
    {
        printf("Graph constructor exited with an error\n");
        pmsis_exit(-5);
    }
#ifdef USE_CAMERA
    pi_camera_control(&cam, PI_CAMERA_CMD_STOP, 0);
    pi_task_callback(&cam_task, cam_handler, NULL);
    pi_camera_capture_async(&cam, Input_1, CAMERA_WIDTH * CAMERA_HEIGHT, &cam_task);
    pi_camera_control(&cam, PI_CAMERA_CMD_START, 0);
    pi_task_wait_on(&cam_task);
#else
    printf("Call cluster\n");
    pi_cluster_send_task_to_cl(&cluster_dev, task);
    resolve_output();
#endif
    __PREFIX(CNN_Destruct)();


    // Close the cluster
    pi_cluster_close(&cluster_dev);
    pi_l2_free(task, (sizeof(struct pi_cluster_task)));
    pi_l2_free(Output_1, (NUM_CLASSES*sizeof(signed short)));
#ifdef USE_CAMERA
    pi_l2_free(Input_1, ((CAMERA_HEIGHT*CAMERA_WIDTH)*sizeof(unsigned char)));
#else
    pi_l2_free(Input_1, ((IMG_H*IMG_W*IMG_C)*sizeof(unsigned char)));
#endif

    pmsis_exit(0);
    return 0;
}



int main(void)
{
    return pmsis_kickoff((void *)application);
}