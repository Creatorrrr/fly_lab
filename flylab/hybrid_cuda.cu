// Six-oscillator controller and reflexes on device. No fast math or FMA.
// One thread owns each world and sums its contacts in their native order.
extern "C" __global__ void hybrid_control(
    double* state, const double* drive, const double* neutral,
    const double* knots, const double* coeff, const double* vectors,
    const double* swing, const double* phase_knots, const double* network,
    const double* cfg, const int* order, const int* actids, const int* adhesionids,
    const int* tips, const int* mapping, const unsigned char* ground,
    const float* xpos, const float* xmat, const int* nacon,
    const int* contact_world, const int* geom, const float* force,
    float* ctrl, float* external_force, double* push_left, const double* push,
    const int* mode, unsigned long long* ticks, float* time,
    int worlds, int nb, int nu, int thorax, int intervals, int maxcon, double t0) {
    int w=blockDim.x*blockIdx.x+threadIdx.x;
    if(w>=worlds)return;
    double* s=state+30*w;
    double* remaining=push_left+w;
    float* frc=external_force+6*nb*w;
    for(int i=0;i<6*nb;i++)frc[i]=0;
    if(*remaining>0){for(int j=0;j<3;j++)frc[thorax*6+j]=push[3*w+j];*remaining=fmax(0.,*remaining-.0001);}
    time[w]=(float)(t0+ticks[w]*.0001);ticks[w]++;
    if(mode[w]==1)return; // Explicit joint targets are already on device.
    double ds[2]={drive[2*w],drive[2*w+1]};
    bool parked=fmax(fabs(ds[0]),fabs(ds[1]))<1.e-7;
    double angle[42],adh[6];
    if(parked){
        for(int i=0;i<42;i++)angle[i]=neutral[i];
        for(int i=0;i<6;i++)adh[i]=cfg[9];
    }else{
        double contact[54]={0.};
        for(int j=0;j<min(nacon[0],maxcon);j++){
            if(contact_world[j]!=w)continue;
            int g1=geom[2*j],g2=geom[2*j+1];
            if(g1<0||g2<0)continue;
            int a=mapping[g1],b=mapping[g2];
            if(a>=0&&ground[g2])for(int k=0;k<3;k++)contact[3*a+k]-=(double)force[6*j+k];
            if(b>=0&&ground[g1])for(int k=0;k<3;k++)contact[3*b+k]+=(double)force[6*j+k];
        }
        double height[6];int sorted[6];
        for(int i=0;i<6;i++){height[i]=(double)xpos[(w*nb+thorax)*3+2]-(double)xpos[(w*nb+tips[i])*3+2];sorted[i]=i;}
        for(int i=1;i<6;i++){int v=sorted[i],j=i-1;while(j>=0&&height[sorted[j]]>height[v]){sorted[j+1]=sorted[j];j--;}sorted[j+1]=v;}
        int selected=height[sorted[5]]>height[sorted[3]]+cfg[0]?sorted[5]:-1;
        if(selected>=0&&s[12+selected]>cfg[7])s[24+selected]=1;
        double phases[6],magnitudes[6];
        for(int i=0;i<6;i++){
            if(s[24+i]>0)s[24+i]++;
            if(s[24+i]>cfg[8])s[24+i]=0;
            double derivative=0.;
            for(int j=0;j<6;j++)derivative+=s[6+j]*network[12+i*6+j]*sin(s[j]-s[i]-network[48+i*6+j]);
            derivative+=6.283185307179586476925286766559*network[i]*(ds[i/3]>=0?1.:-1.);
            phases[i]=s[i]+derivative*.0001;
            magnitudes[i]=s[6+i]+network[6+i]*(fabs(ds[i/3])-s[6+i])*.0001;
        }
        for(int i=0;i<6;i++){
            s[i]=phases[i];s[6+i]=magnitudes[i];
            bool stumble=false;
            for(int j=0;j<3;j++){
                double projection=0.;
                for(int k=0;k<3;k++)projection+=contact[(i*3+j)*3+k]*(double)xmat[(w*nb+thorax)*9+k*3];
                if(projection<cfg[1])stumble=true;
            }
            s[12+i]=(i==selected||s[24+i]>0)?s[12+i]+cfg[2]*.0001:fmax(0.,s[12+i]-cfg[3]*.0001);
            s[18+i]=stumble?s[18+i]+cfg[4]*.0001:fmax(0.,s[18+i]-cfg[5]*.0001);
            double correction=s[12+i]>0?s[12+i]:s[18+i];if(s[12+i]>0)s[18+i]=0.;
            correction=fmin(fmax(0.,correction),cfg[6]);
            double phase=fmod(phases[i],6.283185307179586476925286766559);if(phase<0)phase+=6.283185307179586476925286766559;
            const double values[5]={0.,.8,0.,-.1,0.};double gain=values[0];
            const double* pk=phase_knots+5*i;
            if(phase>=pk[4])gain=values[4];
            else for(int j=0;j<4;j++)if(phase>=pk[j]&&phase<pk[j+1])gain=values[j]+(values[j+1]-values[j])*(phase-pk[j])/(pk[j+1]-pk[j]);
            double wrapped=knots[0]+fmod(phases[i]-knots[0],knots[intervals]-knots[0]);
            if(wrapped<knots[0])wrapped+=knots[intervals]-knots[0];
            int index=0;while(index<intervals-1&&knots[index+1]<=wrapped)index++;
            double dx=wrapped-knots[index];
            for(int j=0;j<7;j++){
                double value=0.,power=1.;
                for(int k=0;k<4;k++){value+=coeff[((i*4+3-k)*intervals+index)*7+j]*power;if(k<3)power*=dx;}
                angle[i*7+j]=neutral[i*7+j]+magnitudes[i]*(value-neutral[i*7+j])+correction*gain*vectors[i*7+j];
            }
            adh[i]=cfg[9]&&!(swing[2*i]<phase&&phase<swing[2*i+1]);
        }
    }
    for(int j=0;j<42;j++)ctrl[w*nu+actids[j]]=(float)angle[order[j]];
    for(int j=0;j<6;j++)ctrl[w*nu+adhesionids[j]]=(float)adh[j];
}

extern "C" __global__ void audit_physics(const float* q,const float* v,
    const int* contacts,const int* collisions,const int* constraints,int* health,
    int worlds,int nq,int nv,int maxcon,int maxefc){
    int w=blockIdx.x*blockDim.x+threadIdx.x;if(w>=worlds)return;
    atomicMax(health+1,contacts[0]);atomicMax(health+2,collisions[0]);atomicMax(health+3,constraints[w]);
    if(contacts[0]>maxcon||collisions[0]>maxcon||constraints[w]>maxefc)atomicExch(health,1);
    for(int j=0;j<nq;j++)if(!isfinite(q[w*nq+j]))atomicExch(health,2);
    for(int j=0;j<nv;j++)if(!isfinite(v[w*nv+j]))atomicExch(health,2);
}
