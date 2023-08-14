extern "C" __device__ unsigned char ray_cube_intersection_cuda(float orig0,
                                                               float orig1,
                                                               float orig2,
                                                               float bounds0_min,
                                                               float bounds1_min,
                                                               float bounds2_min,
                                                               float bounds0_max,
                                                               float bounds1_max,
                                                               float bounds2_max,
                                                               float rdir0,
                                                               float rdir1,
                                                               float rdir2,
                                                               float* t1,
                                                               float* t2){
  // the inverse of the directional vector
  // using the inverse of the directional vector and IEEE floating point arith standard 754
  // makes sure that 0's in the directional vector are handled correctly 
  float invdir0 = 1.f/rdir0;
  float invdir1 = 1.f/rdir1;
  float invdir2 = 1.f/rdir2;
  
  unsigned char intersec = 1;
  
  float t11, t12, t21, t22; 

  if (invdir0 >= 0){
    *t1  = (bounds0_min - orig0) * invdir0;
    *t2  = (bounds0_max - orig0) * invdir0; 
  }
  else{
    *t1  = (bounds0_max - orig0) * invdir0;
    *t2  = (bounds0_min - orig0) * invdir0;
  }
  
  if (invdir1 >= 0){
    t11 = (bounds1_min - orig1) * invdir1; 
    t12 = (bounds1_max - orig1) * invdir1; 
  }
  else{
    t11 = (bounds1_max - orig1) * invdir1;
    t12 = (bounds1_min - orig1) * invdir1; 
  }
  
  if ((*t1 > t12) || (t11 > *t2)){intersec = 0;}
  if (t11 > *t1){*t1 = t11;}
  if (t12 < *t2){*t2 = t12;}
  
  if (invdir2 >= 0){
    t21 = (bounds2_min - orig2) * invdir2; 
    t22 = (bounds2_max - orig2) * invdir2;
  } 
  else{
    t21 = (bounds2_max - orig2) * invdir2; 
    t22 = (bounds2_min - orig2) * invdir2;
  } 
  
  if ((*t1 > t22) || (t21 > *t2)){intersec = 0;}
  if (t21 > *t1){*t1 = t21;}
  if (t22 < *t2){*t2 = t22;} 

  return(intersec);
}

extern "C" __global__ void compute_systemG_kernel(float *xstart, 
                                                      float *xend, 
                                                      float *img_origin, 
                                                      float *voxsize, 
                                                      float *sysG,
                                                      long long nlors, 
                                                      int *img_dim,
                                                      float tofbin_width,
                                                      float *sigma_tof,
                                                      float *tofcenter_offset,
                                                      float n_sigmas,
                                                      short *tof_bin,
                                                      short *nevents)
{
  long long i = blockDim.x * blockIdx.x + threadIdx.x;

  int n0 = img_dim[0];
  int n1 = img_dim[1];
  int n2 = img_dim[2];

  long long nvox = n0 * n1 * n2;

  if(i < nlors)
  {
    float d0, d1, d2, d0_sq, d1_sq, d2_sq; 
    float lsq, cos0_sq, cos1_sq, cos2_sq;
    unsigned short direction; 
    int i0, i1, i2;
    int i0_floor, i1_floor, i2_floor;
    int i0_ceil, i1_ceil, i2_ceil;
    float x_pr0, x_pr1, x_pr2;
    float tmp_0, tmp_1, tmp_2;
   
    float u0, u1, u2, d_norm;
    float x_m0, x_m1, x_m2;    
    float x_v0, x_v1, x_v2;    

    int   it = tof_bin[i];
    float dtof, tw;

    // correction factor for cos(theta) and voxsize
    float cf;

    float sig_tof   = sigma_tof[0];
    float tc_offset = tofcenter_offset[0];

    float xstart0 = xstart[i*3 + 0];
    float xstart1 = xstart[i*3 + 1];
    float xstart2 = xstart[i*3 + 2];

    float xend0 = xend[i*3 + 0];
    float xend1 = xend[i*3 + 1];
    float xend2 = xend[i*3 + 2];

    float voxsize0 = voxsize[0];
    float voxsize1 = voxsize[1];
    float voxsize2 = voxsize[2];

    float img_origin0 = img_origin[0];
    float img_origin1 = img_origin[1];
    float img_origin2 = img_origin[2];

    float neve = (float)nevents[i];

    unsigned char intersec;
    float t1, t2;
    float istart_f, iend_f, tmp;
    int   istart, iend;
    float istart_tof_f, iend_tof_f;
    int   istart_tof, iend_tof;

    // test whether the ray between the two detectors is most parallel
    // with the 0, 1, or 2 axis
    d0 = xend0 - xstart0;
    d1 = xend1 - xstart1;
    d2 = xend2 - xstart2;

    //-----------
    //--- test whether ray and cube intersect
    intersec = ray_cube_intersection_cuda(xstart0, xstart1, xstart2, 
                                          img_origin0 - 1*voxsize0, img_origin1 - 1*voxsize1, img_origin2 - 1*voxsize2,
                                          img_origin0 + n0*voxsize0, img_origin1 + n1*voxsize1, img_origin2 + n2*voxsize2,
                                          d0, d1, d2, &t1, &t2);

    if (intersec == 1)
    {
      d0_sq = d0*d0;
      d1_sq = d1*d1;
      d2_sq = d2*d2;

      lsq = d0_sq + d1_sq + d2_sq;

      cos0_sq = d0_sq / lsq;
      cos1_sq = d1_sq / lsq;
      cos2_sq = d2_sq / lsq;

      direction = 0;
      if ((cos1_sq >= cos0_sq) && (cos1_sq >= cos2_sq))
      {
        direction = 1;
      }
      else
      {
        if ((cos2_sq >= cos0_sq) && (cos2_sq >= cos1_sq))
        {
          direction = 2;
        }
      }
 
      //---------------------------------------------------------
      //--- calculate TOF related quantities
      
      // unit vector (u0,u1,u2) that points from xstart to end
      d_norm = sqrtf(lsq);
      u0 = d0 / d_norm; 
      u1 = d1 / d_norm; 
      u2 = d2 / d_norm; 

      // calculate mid point of LOR
      x_m0 = 0.5f*(xstart0 + xend0);
      x_m1 = 0.5f*(xstart1 + xend1);
      x_m2 = 0.5f*(xstart2 + xend2);

      //---------------------------------------------------------

      if (direction == 0)
      {
        cf = voxsize0 / sqrtf(cos0_sq);

        // case where ray is most parallel to the 0 axis
        // we step through the volume along the 0 direction

        //--- check where ray enters / leaves cube
        istart_f = (xstart0 + t1*d0 - img_origin0) / voxsize0;
        iend_f   = (xstart0 + t2*d0 - img_origin0) / voxsize0;

        if (istart_f > iend_f){
          tmp      = iend_f;
          iend_f   = istart_f;
          istart_f = tmp;
        }
    
        istart = (int)floorf(istart_f);
        iend   = (int)ceilf(iend_f);

        // check in which "plane" the start and end points are
        // we have to do this to avoid that we include voxels
        // that are "outside" the line segment bewteen xstart and xend
        
        // !! for these calculations we overwrite the istart_f and iend_f variables !!
        istart_f = (xstart0 - img_origin0) / voxsize0;
        iend_f   = (xend0   - img_origin0) / voxsize0;

        if (istart_f > iend_f){
          tmp      = iend_f;
          iend_f   = istart_f;
          istart_f = tmp;
        }

        if (istart < (int)floorf(istart_f)){istart = (int)floorf(istart_f);}
        if (iend >= (int)ceilf(iend_f)){iend = (int)ceilf(iend_f);}

        //-- check where we should start and stop according to the TOF kernel
        //-- the tof weights outside +- 3 sigma will be close to 0 so we can
        //-- ignore them         
        istart_tof_f = (x_m0 + (it*tofbin_width - n_sigmas*sig_tof)*u0 - img_origin0) / voxsize0;
        iend_tof_f   = (x_m0 + (it*tofbin_width + n_sigmas*sig_tof)*u0 - img_origin0) / voxsize0;
        
        if (istart_tof_f > iend_tof_f){
          tmp        = iend_tof_f;
          iend_tof_f = istart_tof_f;
          istart_tof_f = tmp;
        }

        istart_tof = (int)floorf(istart_tof_f);
        iend_tof   = (int)ceilf(iend_tof_f);

        if(istart_tof > istart){istart = istart_tof;}
        if(iend_tof   < iend){iend = iend_tof;}
        //-----------
        
        if (istart < 0){istart = 0;}
        if (iend >= n0){iend = n0;}

        for(i0 = istart; i0 < iend; i0++)
        {
          // get the indices where the ray intersects the image plane
          x_pr1 = xstart1 + (img_origin0 + i0*voxsize0 - xstart0)*d1 / d0;    //0轴走到i0坐标时，1轴走到的真实位置
          x_pr2 = xstart2 + (img_origin1 + i0*voxsize1 - xstart0)*d2 / d0;    //2轴真实位置
  
          i1_floor = (int)floorf((x_pr1 - img_origin1)/voxsize1);
          i1_ceil  = i1_floor + 1;
    
          i2_floor = (int)floorf((x_pr2 - img_origin2)/voxsize2);
          i2_ceil  = i2_floor + 1; 
 
          // calculate the distances to the floor normalized to [0,1]
          // for the bilinear interpolation
          tmp_1 = (x_pr1 - (i1_floor*voxsize1 + img_origin1)) / voxsize1;
          tmp_2 = (x_pr2 - (i2_floor*voxsize2 + img_origin2)) / voxsize2;


          //--------- TOF related quantities
          // calculate the voxel center needed for TOF weights
          x_v0 = img_origin0 + i0*voxsize0;
          x_v1 = x_pr1;
          x_v2 = x_pr2;


          // calculate distance of voxel to tof bin center
          dtof = sqrtf(powf((x_m0 + (it*tofbin_width + tc_offset)*u0 - x_v0), 2) + 
                        powf((x_m1 + (it*tofbin_width + tc_offset)*u1 - x_v1), 2) + 
                        powf((x_m2 + (it*tofbin_width + tc_offset)*u2 - x_v2), 2));

          //calculate the TOF weight
          tw = 0.5f*(erff((dtof + 0.5f*tofbin_width)/(sqrtf(2)*sig_tof)) - 
                    erff((dtof - 0.5f*tofbin_width)/(sqrtf(2)*sig_tof)));


          if ((i1_floor >= 0) && (i1_floor < n1) && (i2_floor >= 0) && (i2_floor < n2))
          {
            sysG[i*nvox + n1*n2*i0 + n2*i1_floor + i2_floor] += tw * cf * (1 - tmp_1) * (1 - tmp_2);
          }
          if ((i1_ceil >= 0) && (i1_ceil < n1) && (i2_floor >= 0) && (i2_floor < n2))
          {
            sysG[i*nvox + n1*n2*i0 + n2*i1_ceil + i2_floor] += tw * cf * tmp_1 * (1 - tmp_2);
          }
          if ((i1_floor >= 0) && (i1_floor < n1) && (i2_ceil >= 0) && (i2_ceil < n2))
          {
            sysG[i*nvox + n1*n2*i0 + n2*i1_floor + i2_ceil] += tw * cf * (1 - tmp_1) * tmp_2;
          }
          if ((i1_ceil >= 0) && (i1_ceil < n1) && (i2_ceil >= 0) && (i2_ceil < n2))
          {
            sysG[i*nvox + n1*n2*i0 + n2*i1_ceil + i2_ceil] += tw * cf * tmp_1 * tmp_2;
          }

        }
      }

      //--------------------------------------------------------------------------------- 
      if (direction == 1)
      {
        cf = voxsize1 / sqrtf(cos1_sq);

        // case where ray is most parallel to the 1 axis
        // we step through the volume along the 1 direction

        //--- check where ray enters / leaves cube
        istart_f = (xstart1 + t1*d1 - img_origin1) / voxsize1;
        iend_f   = (xstart1 + t2*d1 - img_origin1) / voxsize1;

        if (istart_f > iend_f){
          tmp      = iend_f;
          iend_f   = istart_f;
          istart_f = tmp;
        }
    
        istart = (int)floorf(istart_f);
        iend   = (int)ceilf(iend_f);

        // check in which "plane" the start and end points are
        // we have to do this to avoid that we include voxels
        // that are "outside" the line segment bewteen xstart and xend
        
        // !! for these calculations we overwrite the istart_f and iend_f variables !!
        istart_f = (xstart1 - img_origin1) / voxsize1;
        iend_f   = (xend1   - img_origin1) / voxsize1;

        if (istart_f > iend_f){
          tmp      = iend_f;
          iend_f   = istart_f;
          istart_f = tmp;
        }

        if (istart < (int)floorf(istart_f)){istart = (int)floorf(istart_f);}
        if (iend >= (int)ceilf(iend_f)){iend = (int)ceilf(iend_f);}

        //-- check where we should start and stop according to the TOF kernel
        //-- the tof weights outside +- 3 sigma will be close to 0 so we can
        //-- ignore them         
        istart_tof_f = (x_m1 + (it*tofbin_width - n_sigmas*sig_tof)*u1 - img_origin1) / voxsize1;
        iend_tof_f   = (x_m1 + (it*tofbin_width + n_sigmas*sig_tof)*u1 - img_origin1) / voxsize1;
        
        if (istart_tof_f > iend_tof_f){
          tmp        = iend_tof_f;
          iend_tof_f = istart_tof_f;
          istart_tof_f = tmp;
        }

        istart_tof = (int)floorf(istart_tof_f);
        iend_tof   = (int)ceilf(iend_tof_f);

        if(istart_tof > istart){istart = istart_tof;}
        if(iend_tof   < iend){iend = iend_tof;}
        //-----------

        if (istart < 0){istart = 0;}
        if (iend >= n1){iend = n1;}
        //---

        for (i1 = istart; i1 < iend; i1++)
        {
          // get the indices where the ray intersects the image plane
          x_pr0 = xstart0 + (img_origin1 + i1*voxsize1 - xstart1)*d0 / d1;
          x_pr2 = xstart2 + (img_origin1 + i1*voxsize1 - xstart1)*d2 / d1;
  
          i0_floor = (int)floorf((x_pr0 - img_origin0)/voxsize0);
          i0_ceil  = i0_floor + 1; 
  
          i2_floor = (int)floorf((x_pr2 - img_origin2)/voxsize2);
          i2_ceil  = i2_floor + 1;

          // calculate the distances to the floor normalized to [0,1]
          // for the bilinear interpolation
          tmp_0 = (x_pr0 - (i0_floor*voxsize0 + img_origin0)) / voxsize0;
          tmp_2 = (x_pr2 - (i2_floor*voxsize2 + img_origin2)) / voxsize2;

          //--------- TOF related quantities
          // calculate the voxel center needed for TOF weights
          x_v0 = x_pr0;
          x_v1 = img_origin1 + i1*voxsize1;
          x_v2 = x_pr2;

          // calculate distance of voxel to tof bin center
          dtof = sqrtf(powf((x_m0 + (it*tofbin_width + tc_offset)*u0 - x_v0), 2) + 
                        powf((x_m1 + (it*tofbin_width + tc_offset)*u1 - x_v1), 2) + 
                        powf((x_m2 + (it*tofbin_width + tc_offset)*u2 - x_v2), 2));

          //calculate the TOF weight
          tw = 0.5f*(erff((dtof + 0.5f*tofbin_width)/(sqrtf(2)*sig_tof)) - 
                    erff((dtof - 0.5f*tofbin_width)/(sqrtf(2)*sig_tof)));


          if ((i0_floor >= 0) && (i0_floor < n0) && (i2_floor >= 0) && (i2_floor < n2))
          {
            sysG[i*nvox + n1*n2*i0_floor + n2*i1 + i2_floor] += tw * cf * (1 - tmp_0) * (1 - tmp_2);
          }
          if ((i0_ceil >= 0) && (i0_ceil < n0) && (i2_floor >= 0) && (i2_floor < n2))
          {
            sysG[i*nvox + n1*n2*i0_ceil + n2*i1 + i2_floor] += tw * cf * tmp_0 * (1 - tmp_2);
          }
          if ((i0_floor >= 0) && (i0_floor < n0) && (i2_ceil >= 0) && (i2_ceil < n2))
          {
            sysG[i*nvox + n1*n2*i0_floor + n2*i1 + i2_ceil] += tw * cf * (1 - tmp_0) * tmp_2;
          }
          if ((i0_ceil >= 0) && (i0_ceil < n0) && (i2_ceil >= 0) && (i2_ceil < n2))
          {
            sysG[i*nvox + n1*n2*i0_ceil + n2*i1 + i2_ceil] += tw * cf * tmp_0 * tmp_2;
          }
        }
      }

      //--------------------------------------------------------------------------------- 
      if (direction == 2)
      {
        cf = voxsize2 / sqrtf(cos2_sq);

        // case where ray is most parallel to the 2 axis
        // we step through the volume along the 2 direction

        //--- check where ray enters / leaves cube
        istart_f = (xstart2 + t1*d2 - img_origin2) / voxsize2;
        iend_f   = (xstart2 + t2*d2 - img_origin2) / voxsize2;

        if (istart_f > iend_f){
          tmp      = iend_f;
          iend_f   = istart_f;
          istart_f = tmp;
        }
    
        istart = (int)floorf(istart_f);
        iend   = (int)ceilf(iend_f);

        // check in which "plane" the start and end points are
        // we have to do this to avoid that we include voxels
        // that are "outside" the line segment bewteen xstart and xend
        
        // !! for these calculations we overwrite the istart_f and iend_f variables !!
        istart_f = (xstart2 - img_origin2) / voxsize2;
        iend_f   = (xend2   - img_origin2) / voxsize2;

        if (istart_f > iend_f){
          tmp      = iend_f;
          iend_f   = istart_f;
          istart_f = tmp;
        }

        if (istart < (int)floorf(istart_f)){istart = (int)floorf(istart_f);}
        if (iend >= (int)ceilf(iend_f)){iend = (int)ceilf(iend_f);}

        //-- check where we should start and stop according to the TOF kernel
        //-- the tof weights outside +- 3 sigma will be close to 0 so we can
        //-- ignore them         
        istart_tof_f = (x_m2 + (it*tofbin_width - n_sigmas*sig_tof)*u2 - img_origin2) / voxsize2;
        iend_tof_f   = (x_m2 + (it*tofbin_width + n_sigmas*sig_tof)*u2 - img_origin2) / voxsize2;
        
        if (istart_tof_f > iend_tof_f){
          tmp        = iend_tof_f;
          iend_tof_f = istart_tof_f;
          istart_tof_f = tmp;
        }

        istart_tof = (int)floorf(istart_tof_f);
        iend_tof   = (int)ceilf(iend_tof_f);

        if(istart_tof > istart){istart = istart_tof;}
        if(iend_tof   < iend){iend = iend_tof;}
        //-----------

        if (istart < 0){istart = 0;}
        if (iend >= n2){iend = n2;}
        //---

        for(i2 = istart; i2 < iend; i2++)
        {
          // get the indices where the ray intersects the image plane
          x_pr0 = xstart0 + (img_origin2 + i2*voxsize2 - xstart2)*d0 / d2;
          x_pr1 = xstart1 + (img_origin2 + i2*voxsize2 - xstart2)*d1 / d2;
  
          i0_floor = (int)floorf((x_pr0 - img_origin0)/voxsize0);
          i0_ceil  = i0_floor + 1;
  
          i1_floor = (int)floorf((x_pr1 - img_origin1)/voxsize1);
          i1_ceil  = i1_floor + 1; 
  
          // calculate the distances to the floor normalized to [0,1]
          // for the bilinear interpolation
          tmp_0 = (x_pr0 - (i0_floor*voxsize0 + img_origin0)) / voxsize0;
          tmp_1 = (x_pr1 - (i1_floor*voxsize1 + img_origin1)) / voxsize1;

          //--------- TOF related quantities
          // calculate the voxel center needed for TOF weights
          x_v0 = x_pr0;
          x_v1 = x_pr1;
          x_v2 = img_origin2 + i2*voxsize2;


          // calculate distance of voxel to tof bin center
          dtof = sqrtf(powf((x_m0 + (it*tofbin_width + tc_offset)*u0 - x_v0), 2) + 
                        powf((x_m1 + (it*tofbin_width + tc_offset)*u1 - x_v1), 2) + 
                        powf((x_m2 + (it*tofbin_width + tc_offset)*u2 - x_v2), 2));

          //calculate the TOF weight
          tw = 0.5f*(erff((dtof + 0.5f*tofbin_width)/(sqrtf(2)*sig_tof)) - 
                    erff((dtof - 0.5f*tofbin_width)/(sqrtf(2)*sig_tof)));


          if ((i0_floor >= 0) && (i0_floor < n0) && (i1_floor >= 0) && (i1_floor < n1))
          {
            sysG[i*nvox + n1*n2*i0_floor + n2*i1_floor + i2] += tw * cf * (1 - tmp_0) * (1 - tmp_1);
          }
          if ((i0_ceil >= 0) && (i0_ceil < n0) && (i1_floor >= 0) && (i1_floor < n1))
          {
            sysG[i*nvox + n1*n2*i0_ceil + n2*i1_floor + i2] += tw* cf * tmp_0 * (1 - tmp_1);
          }
          if ((i0_floor >= 0) && (i0_floor < n0) && (i1_ceil >= 0) & (i1_ceil < n1))
          {
            sysG[i*nvox + n1*n2*i0_floor + n2*i1_ceil + i2] += tw * cf * (1 - tmp_0) * tmp_1;
          }
          if ((i0_ceil >= 0) && (i0_ceil < n0) && (i1_ceil >= 0) && (i1_ceil < n1))
          {
            sysG[i*nvox + n1*n2*i0_ceil + n2*i1_ceil + i2] += tw * cf * tmp_0 * tmp_1;
          }
        }
      }
    }
  }
}
