__kernel void
stencil(__global float *B,
        __global float *A,
        unsigned int line_size)
{
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    const unsigned int xloc = get_local_id(0);
    const unsigned int yloc = get_local_id(1);

    __local float tile[16+2][4+2];

    A += line_size + 16; // OFFSET
    B += line_size + 16; // OFFSET

    // Copy tile of A into local memory
    // Begin with inner values
    for(int k=0; k<4; k++)
    {
        tile[yloc*4+k+1][xloc+1] = A[(y*4+k)*line_size + x];
    }
    // and finish with the borders

    {
        const int cbx = xloc;
        const int cby = (yloc & 1) ? -1 : 16;
        const int bx = (yloc & 2) ? cbx : cby;
        const int by = (yloc & 2) ? cby : cbx;
        //TODO trouver les bons indices...
        tile[cbx][cby] = A[];

    }

    barrier(CLK_LOCAL_MEM_FENCE);
    int tmp = y*4;
    for(int k=0; k<4; k++)
        B[(tmp + k)*line_size + x] = 0.75 * tile[x+1][k+1] +
                                     0.25*( tile[x-1+1][k+1] +
                                            tile[x+1+1][k+1] +
                                            tile[x+1][k-1+1] +
                                            tile[x+1][k+1+1]);

}


