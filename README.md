# The problem with walls in Ray Tracer Challenge

How I use the `rt.rs`:

```rust
    let world = crate::rt::test_world_2();
    let mut camera = crate::camera!(100, 50, crate::degree!(60));
    *camera.get_transform_matrix_mut() = crate::rt::view_transform(
        crate::point!(0, 1.5, -5),
        crate::point!(0, 1, 0),
        crate::normalised_direction!(0, 1, 0),
    );
    camera.pixel_width = 500;
    camera.pixel_height = 500;

    let mut ppm = format!(
        "P3 {} {} 255\n",
        camera.pixel_width, camera.pixel_height
    );

    let colors = self.camera.render(&self.world);
    colors.into_iter().flatten().for_each(|color| {
        ppm.push_str(&format!(
            "{} {} {} ",
            (255.0 * color.r()) as u8,
            (255.0 * color.g()) as u8,
            (255.0 * color.b()) as u8
        ));
    });
    std::fs::write("/tmp/rt.ppm", ppm).expect("Couldn't write data");
```
