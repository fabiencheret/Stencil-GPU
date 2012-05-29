__kernel void
stencil(__global float *B,
        __global float *A,
        unsigned int line_size)
{
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    const unsigned int xloc = get_local_id(0);
    const unsigned int yloc = get_local_id(1);

    __local float tile[16+2][16+2];

    A += line_size + 16; // OFFSET
    B += line_size + 16; // OFFSET

    // Copy tile of A into local memory
    // Begin with inner values
    for(int k=0; k<4; k++)
    {
        tile[xloc+1][yloc*4+k+1] = A[(y*4+k)*line_size + x];
    }
    // and finish with the borders

    {
        const int cbx = xloc;
        const int cby = (yloc & 1) ? -1 : 16;
        const int bx = (yloc & 2) ? cbx : cby;
        const int by = (yloc & 2) ? cby : cbx;
        //TODO trouver les bons indices...
        tile[cbx+1][yloc*4+cby+1] = A[(y*4+cby)*line_size + x];
        tile[bx+1][yloc*4+by+1] = A[(y*4+by)*line_size + x];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    int tmp = y*4;
    for(int k=0; k<4; k++)
        B[(tmp + k)*line_size + x] = 0.75 * tile[xloc+1][yloc*4+k+1] +
                                     0.25*( tile[xloc-1+1][yloc*4+k+1] +
                                            tile[xloc+1+1][yloc*4+k+1] +
                                            tile[xloc+1][yloc*4+k-1+1] +
                                            tile[xloc+1][yloc*4+k+1+1]);

}


