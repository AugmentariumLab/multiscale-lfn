<style>
@media screen and (min-width: 64em) {
.main-content {
    max-width: 70rem;
}
}
.page-header{
background-color: #086375;
background-image: linear-gradient(120deg, #156064, #086375);
}
.erp_image {
    width: 12.4rem;
    object-fit: fill;
}
.main-content table th, .main-content table td {
    padding: 0;
}
.table_header td {
  text-align: center;
}
.comparison_table {
  border: 1px solid;
}
</style>

<div class='container'>
    <h2 id="Examples" style="padding-top: 80px; margin-top: -80px">Multiple LFNs vs Multi-scale LFN</h2>
    <p>
        Instead of utilizing multiple LFNs for antialiasing, our multi-scale LFN achieves anti-aliasing using only a single model.
    </p>
    <table class="comparison_table" cellspacing="3">
        <tr class="table_header">
        <td>
            Model
        </td>
        <td>
            LOD 1 (1/8 scale)
        </td>
        <td>
            LOD 2 (1/4 scale)
        </td>
        <td>
            LOD 3 (1/2 scale)
        </td>
        <td>
            LOD 4 (1/1 scale)
        </td>
        </tr>
        <tr>
        <td>Multiple LFNs</td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset1/multiple_r8.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset1/multiple_r4.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset1/multiple_r2.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset1/fullscale_r1.mp4" type="video/mp4">
        </video></td>
        </tr>
        <tr>
        <td>Multi-scale LFN</td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset1/multiscale_r8.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset1/multiscale_r4.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset1/multiscale_r2.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset1/multiscale_r1.mp4" type="video/mp4">
        </video></td>
        </tr>
        <tr>
        <td>Multiple LFNs</td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset2/multiple_r8.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset2/multiple_r4.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset2/multiple_r2.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset2/fullscale_r1.mp4" type="video/mp4">
        </video></td>
        </tr>
        <tr>
        <td>Multi-scale LFN</td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset2/multiscale_r8.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset2/multiscale_r4.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset2/multiscale_r2.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset2/multiscale_r1.mp4" type="video/mp4">
        </video></td>
        </tr>
        <tr>
        <td>Multiple LFNs</td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset3/multiple_r8.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset3/multiple_r4.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset3/multiple_r2.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset3/fullscale_r1.mp4" type="video/mp4">
        </video></td>
        </tr>
        <tr>
        <td>Multi-scale LFN</td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset3/multiscale_r8.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset3/multiscale_r4.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset3/multiscale_r2.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset3/multiscale_r1.mp4" type="video/mp4">
        </video></td>
        </tr>
        <tr>
        <td>Multiple LFNs</td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset4/multiple_r8.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset4/multiple_r4.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset4/multiple_r2.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset4/fullscale_r1.mp4" type="video/mp4">
        </video></td>
        </tr>
        <tr>
        <td>Multi-scale LFN</td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset4/multiscale_r8.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset4/multiscale_r4.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset4/multiscale_r2.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset4/multiscale_r1.mp4" type="video/mp4">
        </video></td>
        </tr>
        <tr>
        <td>Multiple LFNs</td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset5/multiple_r8.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset5/multiple_r4.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset5/multiple_r2.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset5/fullscale_r1.mp4" type="video/mp4">
        </video></td>
        </tr>
        <tr>
        <td>Multi-scale LFN</td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset5/multiscale_r8.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset5/multiscale_r4.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset5/multiscale_r2.mp4" type="video/mp4">
        </video></td>
        <td><video muted autoplay loop class="erp_image">
            <source src="resources/videos/dataset5/multiscale_r1.mp4" type="video/mp4">
        </video></td>
        </tr>
    </table>
</div>
