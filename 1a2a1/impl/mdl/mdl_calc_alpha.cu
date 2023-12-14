#include "mdl.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

float mdl_moy_dp(Mdl_t * mdl, uint c) {
	float * dp = gpu_vers_cpu<float>(mdl->dp__d[c], (mdl->ST[c-1]+1)*mdl->ST[c]);
	float moy = 0;
	FOR(0, i, mdl->ST[c]*(mdl->ST[c-1]+1)) {
		moy += fabs(dp[i]);
	};
	free(dp);
	return moy / (float)(mdl->ST[c]*(mdl->ST[c-1]+1));
}