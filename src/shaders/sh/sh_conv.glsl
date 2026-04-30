void sh_conv(float f[16], float g[16], out float result[16]){
	result[0] = 3.544908 * f[0] * g[0];
	result[1] = 2.046653 * f[1] * g[2];
	result[2] = 2.046653 * f[2] * g[2];
	result[3] = 2.046653 * f[3] * g[2];
	result[4] = 1.585331 * f[4] * g[6];
	result[5] = 1.585331 * f[5] * g[6];
	result[6] = 1.585331 * f[6] * g[6];
	result[7] = 1.585331 * f[7] * g[6];
	result[8] = 1.585331 * f[8] * g[6];
	result[9] = 1.339849 * f[9] * g[12];
	result[10] = 1.339849 * f[10] * g[12];
	result[11] = 1.339849 * f[11] * g[12];
	result[12] = 1.339849 * f[12] * g[12];
	result[13] = 1.339849 * f[13] * g[12];
	result[14] = 1.339849 * f[14] * g[12];
	result[15] = 1.339849 * f[15] * g[12];
}