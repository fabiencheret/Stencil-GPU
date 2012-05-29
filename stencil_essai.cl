__kernel void
stencil(__global float *B,
        __global float *A,
        unsigned int line_size)
{

    __local float tile[18][18];

    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);

    const unsigned int xloc = get_local_id(0);
    const unsigned int yloc = get_local_id(1);

    //x = get_group_id * 16 + xloc
    //y = get_group_id * 16 + yloc

    A += line_size + 16;
    B += line_size + 16;

    //on copie dans la tile les valeurs du milieu

    for(int i=0; i<4; i++)
    {
        tile[xloc+1][yloc*4+i+1] = A[(y*4+i)*line_size + x];
    }

    //reste à copier les bords.
    //on sépare les threads

    const int cbx = xloc;
    const int cby = (yloc & 1) ? -1 : 16;
    const int bx = (yloc & 2) ? cbx : cby;
    const int by = (yloc & 2) ? cby : cbx;
//on copie les bords...
    tile[cbx][cby+1] = A[((y-yloc+cby)*4)*line_size + x];
    for(int i=0; i<4; i++)
    {
        tile[bx+1][by+1+i] = A[((y-yloc+by)*4+i)*line_size + x - xloc + bx];
    }
//on prie...

//boucle
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int k=0; k<4; k++)
        B[(tmp + y*4)*line_size + x] = 0.75 * tile[xloc+1][yloc*4+k+1] +
                                       0.25*( tile[xloc][yloc*4+k+1] +
                                              tile[xloc+2][yloc*4+k+1] +
                                              tile[xloc+1][yloc*4+k] +
                                              tile[xloc+1][yloc*4+k+2]);


}
