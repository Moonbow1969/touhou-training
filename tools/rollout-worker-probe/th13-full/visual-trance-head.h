/* Independent binary head for the TH13-only engineering actor. The 84 input
 * values come solely from the current and preceding 96x112 original camera. */
static float trance_weights[85];
static uint64_t trance_rng;
static FILE *trance_file;
static double trance_p,trance_lp;
static int trance_bit;
static int trance_open(uint64_t seed) {
    FILE *f=fopen("/capture/lib/trance-head.bin","rb");
    char magic[8];
    if(!f)return 0;
    int valid=fread(magic,1,8,f)==8 && !memcmp(magic,"TH13TR01",8) &&
        fread(trance_weights,sizeof(float),85,f)==85 && fgetc(f)==EOF;
    fclose(f);
    if(!valid)return 0;
    for(int i=0;i<85;i++)if(!isfinite(trance_weights[i]))return 0;
    trance_rng=seed^UINT64_C(0xd6e8feb86659fd93);
    trance_file=fopen("fifth-head.csv","w");
    if(!trance_file)return 0;
    fputs("step,sampled_c,probability_c,sampled_logp,requested_full_mask,x11_full_mask,native_full_mask,pre_tick,post_tick\n",trance_file);
    return 1;
}
static int trance_step(const float *features) {
    double logit=trance_weights[0];
    for(int i=0;i<84;i++)logit+=(double)features[i]*trance_weights[i+1];
    trance_p=1./(1.+exp(-logit));
    uint64_t z=(trance_rng+=UINT64_C(0x9e3779b97f4a7c15));
    z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);
    double draw=((z^(z>>31))>>11)*0x1.0p-53;
    trance_bit=draw<trance_p;
    trance_lp=trance_bit?log(trance_p):log1p(-trance_p);
    return trance_bit?0xa00:0;
}
