#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <ve_offload.h>

#define DBGPRT(...)
//#define DBGPRT printf

#define BUF_START	(0xFE)
#define BUF_END		(0xFF)

typedef enum {
    E_MEMTYPE_HOST = 0,
    E_MEMTYPE_VE,
    E_MEMTYPE_MAX
}E_MEMTYPE;

#define  DEV_NAME_LEN (16)
typedef struct {
    char dev1[DEV_NAME_LEN+1];
    char dev2[DEV_NAME_LEN+1];
    long n;
    long loop_count;
    long dev1_node;
    long dev2_node;
}ARGS;

typedef struct {
    unsigned  char* data;
    unsigned long itemsize;
    unsigned long nelem;
    unsigned long size;
    E_MEMTYPE memtype;
}BUF;

struct veo_proc_handle *proc = NULL;

unsigned long min(unsigned long val1, unsigned long val2)
{
    if( val1 < val2) return val1;
    return val2;
}

void send_recv_helper(BUF buf, int rank)
{
    MPI_Status stat;
    int tag1 = 10;
    int tag2 = 20;
    unsigned long max_count = pow(2, 31);
    unsigned long begin = 0;
    unsigned long remain = buf.size;
    while(remain > 0) {
        void* part_buf = (void*)&buf.data[begin];
        unsigned long part_nelem = min(remain / buf.itemsize, max_count - 1);
        unsigned long part_size = part_nelem * buf.itemsize;
        DBGPRT("rank=%d, begin=%lu, part_buf=0x%lx, part_size=%lu, part_nelem=%lu, remain=%lu\n",
               rank, begin, part_buf, part_size, part_nelem, remain);
        if(rank == 0){
            MPI_Send(part_buf, part_nelem, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
            MPI_Recv(part_buf, part_nelem, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, &stat);
        } else if(rank == 1){
            MPI_Recv(part_buf, part_nelem, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
            MPI_Send(part_buf, part_nelem, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
        }
        begin += part_size;
        if( remain <= part_size) break;
        remain -= part_size;
    }
}

void get_args(int argc, char *argv[], ARGS* out)
{
    int ret = -1;
    int option_index = 0;

    struct option long_options[] = {
        {"dev1",       required_argument, NULL, '1' },
        {"dev2",       required_argument, NULL, '2' },
        {"n",          required_argument, NULL, 'n' },
        {"loop_count", required_argument, NULL, 'l' },
        {"dev1_node",  required_argument, NULL, 'x' },
        {"dev2_node2", required_argument, NULL, 'y' },
        {0,            0,                 0,    0 }
    };
    strncpy(out->dev1,"vh",DEV_NAME_LEN);
    out->dev1_node = -1;
    strncpy(out->dev2,"vh",DEV_NAME_LEN);
    out->dev2_node = -1;
    out->n = 20;
    out->loop_count = 10;

    while(1){
        option_index = 0;
        ret = getopt_long(argc, argv, "1:2:n:l:x:y:", long_options, &option_index);
        if (ret == -1)
            break;
        switch(ret){
        case '1':
            DBGPRT("option dev1 = %s \n", optarg);
            strncpy(out->dev1, optarg, DEV_NAME_LEN);
            break;
        case '2':
            DBGPRT("option dev2 = %s \n", optarg);
            strncpy(out->dev2, optarg, DEV_NAME_LEN);
            break;
        case 'n':
            DBGPRT("option n = %s \n", optarg);
            out->n = atol(optarg);
            break;
        case 'l':
            DBGPRT("option loop_count = %s \n", optarg);
            out->loop_count = atol(optarg);
            break;
        case 'x':
            DBGPRT("option dev1_node = %s \n", optarg);
            out->dev1_node = atol(optarg);
            break;
        case 'y':
            DBGPRT("option dev2_node = %s \n", optarg);
            out->dev2_node = atol(optarg);
            break;
        default:
            printf("?? getopt returned character code 0%o ??\n", ret);
        }
    }
}

void print_vars(ARGS args)
{
    printf("{'dev1': '%s', 'dev1_node': %lu, 'dev2': '%s', 'dev2_node': %lu, 'n': %lu, 'loop_count': %lu}\n",
        args.dev1, args.dev1_node, args.dev2, args.dev2_node, args.n, args.loop_count);
}

void mem_alloc( BUF* buf, char* dev, int node, unsigned long nelem, unsigned long itemsize)
{
    int ret = 0;
    buf->itemsize = itemsize;
    buf->nelem = nelem;
    buf->size = itemsize * nelem;
    DBGPRT("mem_alloc start dev=%s, nelem = %lu, itemsize=%lu, size=%lu\n",dev, nelem, itemsize, buf->size);
    if( strcmp(dev,"ve") == 0) {
        if( proc == NULL)
            proc = veo_proc_create( node ); 
        ret = veo_alloc_hmem( proc, (void*)&buf->data, buf->size );
        if (ret != 0) {
            printf("failed veo_alloc_hmem\n");
            MPI_Finalize();
            exit(0);
        }
        buf->memtype = E_MEMTYPE_VE;
    } else if(strcmp(dev,"vh") == 0) {
        buf->data = malloc(buf->size);
        if (buf->data == NULL) {
            printf("failed malloc\n");
            MPI_Finalize();
            exit(0);
        }
        buf->memtype = E_MEMTYPE_HOST;
    } else {
        printf(" unknown DEV(%s) err\n",dev);
        MPI_Finalize();
        exit(-1);
    }
    DBGPRT("mem_alloc end dev=%s data=0x%lx\n",dev, buf->data);
}

void mem_free( BUF buf) 
{
    if( buf.memtype == E_MEMTYPE_VE) {
        veo_free_hmem(buf.data);
    } else if(buf.memtype == E_MEMTYPE_HOST) {
        free(buf.data);
    } else {
        printf(" unknown memtype(%lu) err\n",buf.memtype);
        MPI_Finalize();
        exit(-1);
    }
}

void mem_check_set(BUF buf, int rank)
{
    unsigned char start = BUF_START;
    unsigned char end = BUF_END;
    if( rank == 0 ) {
        if( buf.memtype == E_MEMTYPE_VE) {
            veo_hmemcpy((void*)&buf.data[0], (void*)&start, 1);
            veo_hmemcpy((void*)&buf.data[buf.size-1],(void*)&end,1);
        } else if(buf.memtype == E_MEMTYPE_HOST) {
            memcpy((void*)&buf.data[0], (void*)&start, 1);
            memcpy((void*)&buf.data[buf.size-1], (void*)&end,1);
        } else {
            printf(" unknown memtype(%lu) err\n",buf.memtype);
            MPI_Finalize();
            exit(-1);
        }
        DBGPRT("rank=%d, memcheck set start = 0x%x, end = 0x%x \n", rank, start, end);
    }
}

void mem_check(BUF buf, int rank)
{
    unsigned char start = 0;
    unsigned char end = 0;
    if( buf.memtype == E_MEMTYPE_VE) {
        veo_hmemcpy((void*)&start, (void*)&buf.data[0], 1);
        veo_hmemcpy((void*)&end, (void*)&buf.data[buf.size-1], 1);
    } else if(buf.memtype == E_MEMTYPE_HOST) {
        memcpy((void*)&start, (void*)&buf.data[0], 1);
        memcpy((void*)&end, (void*)&buf.data[buf.size-1],1);
    } else {
        printf(" unknown memtype(%lu) err\n",buf.memtype);
        MPI_Finalize();
        exit(-1);
    }
    DBGPRT("rank=%d, memcheck start = 0x%x, end = 0x%x \n", rank, start, end);
    if( start != BUF_START || end != BUF_END) {
        printf("Result mismatch (rank = %d)",rank);
        MPI_Finalize();
        exit(-1);
    }
}

int main(int argc, char *argv[]) 
{
    int size = 0;
    int rank = 0;
    double t0;
    double t1;
    double elapsed_time;
    double avg_transfer_time;
    double bandwidth;
    ARGS args;
    BUF buf;
    int i;
    int n;
    unsigned long nelem;

    // arg parse.
    get_args(argc, argv, &args);

    // set module
    if(strcmp(args.dev1,"vh") == 0){
        //dev1 = np
    } else if(strcmp(args.dev1, "ve") == 0){
        //dev1 = vp
    } else {
        printf("args.dev1 ValueError\n");
        return -1;
    }
    if(strcmp(args.dev2,"vh") == 0){
        //dev2 = np
    } else if(strcmp(args.dev2, "ve") == 0){
        //dev2 = vp
    } else {
        printf("args.dev2 ValueError\n");
        return -1;
    }
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0)
        print_vars(args);
    if(size != 2){
        printf("size != 2 ValueError\n");
        MPI_Finalize();
        return -1;
    }

    if(rank == 0){
        printf("| Data Size (B) | Avg Transfer Time (s) | Bandwidth (GB/s) |\n");
        printf("|---------------|-----------------------|------------------|\n");
    }

    for( n = 0; n < args.n; n++){ 
        nelem = (unsigned long)1 << (unsigned long)n;
        if(rank == 0){
            mem_alloc(&buf, args.dev1, args.dev1_node, nelem, sizeof(double));
        } else {
            mem_alloc(&buf, args.dev2, args.dev2_node, nelem, sizeof(double));
        }
        mem_check_set(buf, rank);

        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for( i = 0; i < args.loop_count; i++){
            send_recv_helper(buf, rank);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        t1 = MPI_Wtime();
        elapsed_time = t1 - t0;
        avg_transfer_time = elapsed_time / (2.0 * args.loop_count);
        bandwidth = buf.size / (pow(1024, 3)) / avg_transfer_time;
        if(rank == 0)
            printf("|%15ld|%23.9lf|%18.9lf|\n",buf.size, avg_transfer_time, bandwidth);

        mem_check(buf, rank);
        mem_free(buf);
    }
    if(rank == 0){
        printf("|---------------|-----------------------|------------------|\n");
    }
    MPI_Finalize();
    return 0;
}
