//! the ray tracing challenge.
use std::ops::DerefMut;

use derive_builder::Builder;
use glam::Vec4Swizzles;

/// Creates a point.
#[macro_export]
macro_rules! point {
    () => {
        point!(0.0)
    };
    ($coord:expr) => {
        point!($coord, $coord)
    };
    ($x:expr, $y:expr) => {
        point!($x, $y, 0.0)
    };
    ($x:expr, $y:expr, $z:expr) => {
        $crate::rt::Point::new($x as f32, $y as f32, $z as f32)
    };

    (x $x:expr) => {
        point!($x as f32, 0.0)
    };
    (y $y:expr) => {
        point!(0.0, $y as f32)
    };
    (z $z:expr) => {
        point!(0.0, 0.0, $z as f32)
    }; // (x$x: expr) => {
       //     point!($x as f32, 0.0)
       // };
       // (y_$y: expr) => {
       //     point!(0.0, $y as f32)
       // };
       // (z($z:expr)) => {{
       //     point!(0.0, 0.0, $z as f32)
       // }};
}

/// Creates a camera.
#[macro_export]
macro_rules! camera {
    ($pixel_width:expr, $pixel_height:expr, $fov:expr) => {
        $crate::rt::Camera::new(
            $pixel_width as usize,
            $pixel_height as usize,
            $crate::rt::Radians::from($fov),
            glam::Mat4::IDENTITY,
        )
    };
}

/// Calculates an angle in degrees in a little better way.
#[macro_export]
macro_rules! degree {
    ($num:expr) => {
        $crate::rt::angle_from_pi_division(180.0 / ($num as f32))
    };
}

/// Creates a direction.
#[macro_export]
macro_rules! direction {
    ($x: expr, $y: expr, $z: expr) => {
        $crate::rt::Direction::from(glam::vec3($x as f32, $y as f32, $z as f32))
    };
    ($x: expr, $y: expr, $z: expr, $w: expr) => {
        $crate::rt::Direction::from(glam::vec4($x as f32, $y as f32, $z as f32, $w as f32))
    };
}

/// Creates a normalised direction.
#[macro_export]
macro_rules! normalised_direction {
    ($x: expr, $y: expr, $z: expr) => {
        $crate::rt::NormalisedDirection::from(glam::vec4($x as f32, $y as f32, $z as f32, 0.0))
    };
}

/// Creates a color.
#[macro_export]
macro_rules! color {
    ($c: expr) => {
        color!($c, $c, $c, $c)
    };
    ($r: expr, $g: expr, $b: expr) => {
        color!($r as f32, $g as f32, $b as f32, 1.0)
    };
    ($r: expr, $g: expr, $b: expr, $a: expr) => {
        $crate::rt::ColorRGBA(glam::vec4($r as f32, $g as f32, $b as f32, $a as f32))
    };
}

/// Rotates an object.
#[macro_export]
macro_rules! rotate {
    ($obj:expr, x, $angle:expr) => {
        $obj.rotate([$crate::rt::Radians::from($angle), degree!(0), degree!(0)])
    };

    ($obj:expr, y, $angle:expr) => {
        $obj.rotate([degree!(0), $crate::rt::Radians::from($angle), degree!(0)])
    };

    ($obj:expr, z, $angle:expr) => {
        $obj.rotate([degree!(0), degree!(0), $crate::rt::Radians::from($angle)])
    };
}

/// Scales an object.
#[macro_export]
macro_rules! scale {
    ($obj:expr, $x:expr, $y:expr, $z:expr) => {
        $obj.scale(glam::vec3($x as f32, $y as f32, $z as f32))
    };
    ($obj:expr, $amount:expr) => {
        scale!($obj, xyz, $amount)
    };
    ($obj:expr, xyz, $amount:expr) => {
        $obj.scale(glam::vec3($amount as f32, $amount as f32, $amount as f32))
    };
    ($obj:expr, xy, $amount:expr) => {
        $obj.scale(glam::vec3($amount as f32, $amount as f32, 1.0))
    };
    ($obj:expr, x, $amount:expr) => {
        $obj.scale(glam::vec3($amount as f32, 1.0, 1.0))
    };
    ($obj:expr, y, $amount:expr) => {
        $obj.scale(glam::vec3(1.0, $amount as f32, 1.0))
    };
    ($obj:expr, z, $amount:expr) => {
        $obj.scale(glam::vec3(1.0, 1.0, $amount as f32))
    };
}

/// Translates an object.
#[macro_export]
macro_rules! translate {
    ($obj:expr, $x:expr, $y:expr, $z:expr) => {
        $obj.translate(glam::vec3($x as f32, $y as f32, $z as f32))
    };
    ($obj:expr, $amount:expr) => {
        translate!($obj, xyz, $amount)
    };
    ($obj:expr, xyz, $amount:expr) => {
        $obj.translate(glam::vec3($amount as f32, $amount as f32, $amount as f32))
    };
    ($obj:expr, xy, $amount:expr) => {
        $obj.translate(glam::vec3($amount as f32, $amount as f32, 0.0))
    };
    ($obj:expr, x, $amount:expr) => {
        $obj.translate(glam::vec3($amount as f32, 0.0, 0.0))
    };
    ($obj:expr, y, $amount:expr) => {
        $obj.translate(glam::vec3(0.0, $amount as f32, 0.0))
    };
    ($obj:expr, z, $amount:expr) => {
        $obj.translate(glam::vec3(0.0, 0.0, $amount as f32))
    };
}

/// Transform an object using a matrix.
#[macro_export]
macro_rules! transform {
    ($obj:expr, $mat:expr) => {
        *$obj.get_transform_matrix_mut() = $mat;
    };
    ($obj:expr, from $from:expr, to $to:expr, up $up:expr) => {
        let vt = $crate::rt::view_transform($from, $to, $up);
        *$obj.get_transform_matrix_mut() = vt;
    };
    ($obj:expr, view, $from:expr, $to:expr, $up:expr) => {
        let vt = $crate::rt::view_transform($from, $to, $up);
        *$obj.get_transform_matrix_mut() = vt;
    };
}

/// A transformation matrix.
#[derive(Debug, Copy, Clone)]
pub struct TransformationMatrixBuilder {
    /// The scaling matrix.
    scale: glam::Mat4,
    /// The rotation matrix.
    rotation: glam::Mat4,
    /// The translation matrix.
    translation: glam::Mat4,
    ///// Quaternion.
    // quaternion: glam::Quat,
}
impl TransformationMatrixBuilder {
    /// Creates new builder.
    pub fn new() -> Self {
        Self {
            scale: glam::Mat4::IDENTITY,
            rotation: glam::Mat4::IDENTITY,
            translation: glam::Mat4::IDENTITY,
            // quaternion: glam::Quat::IDENTITY,
        }
    }

    /// Add rotation.
    pub fn rotation(mut self, radians: [Radians; 3]) -> Self {
        // self.quaternion = glam::Quat::from_ax
        // self.rotation = glam::Mat4::from_rotation_z(radians[2].0) * glam::Mat4::from_rotation_y(radians[1].0) * glam::Mat4::from_rotation_x(radians[0].0);
        // self.rotation = glam::Mat4::from_rotation_x(radians[0].0) * glam::Mat4::from_rotation_y(radians[1].0) * glam::Mat4::from_rotation_z(radians[2].0);
        self.rotate_x(radians[0]).rotate_y(radians[1]).rotate_z(radians[2])
    }

    /// Rotates by X.
    ///
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    ///
    /// let t = TransformationMatrixBuilder::new()
    ///           .rotate_x(degree!(45))
    ///           .build();
    /// assert_eq!(t * point!(0, 1, 0).0, point!(0, 0.70710677, 0.70710677).0);
    ///
    /// let t = TransformationMatrixBuilder::new()
    ///           .rotate_x(degree!(90))
    ///           .build();
    /// assert_eq!(t * point!(0, 1, 0).0, point!(0, -4.371139e-8, 1).0);
    /// ```
    pub fn rotate_x(mut self, radians: Radians) -> Self {
        self.rotation = self.rotation * glam::mat4(
            glam::vec4(1.0, 0.0, 0.0, 0.0),
            glam::vec4(0.0, radians.0.cos(), radians.0.sin(), 0.0),
            glam::vec4(0.0, -radians.0.sin(), radians.0.cos(), 0.0),
            glam::vec4(0.0, 0.0, 0.0, 1.0),
        );
        self
    }

    /// Rotates by Y.
    ///
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    ///
    /// let t = TransformationMatrixBuilder::new()
    ///           .rotate_y(degree!(45))
    ///           .build();
    /// assert_eq!(t * point!(0, 0, 1).0, point!(0.70710677, 0, 0.70710677).0);
    ///
    /// let t = TransformationMatrixBuilder::new()
    ///           .rotate_y(degree!(90))
    ///           .build();
    /// assert_eq!(t * point!(0, 0, 1).0, point!(1, 0, -4.371139e-8).0);
    /// ```
    pub fn rotate_y(mut self, radians: Radians) -> Self {
        self.rotation = self.rotation * glam::mat4(
            glam::vec4(radians.0.cos(), 0.0, -radians.0.sin(), 0.0),
            glam::vec4(0.0, 1.0, 0.0, 0.0),
            glam::vec4(radians.0.sin(), 0.0, radians.0.cos(), 0.0),
            glam::vec4(0.0, 0.0, 0.0, 1.0),
        );
        self
    }


    /// Rotates by Z.
    ///
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    ///
    /// let t = TransformationMatrixBuilder::new()
    ///           .rotate_z(degree!(45))
    ///           .build();
    /// assert_eq!(t * point!(0, 1, 0).0, point!(-0.70710677, 0.70710677, 0).0);
    ///
    /// let t = TransformationMatrixBuilder::new()
    ///           .rotate_z(degree!(90))
    ///           .build();
    /// assert_eq!(t * point!(0, 1, 0).0, point!(-1, -4.371139e-8, 0).0);
    /// ```
    pub fn rotate_z(mut self, radians: Radians) -> Self {
        self.rotation = self.rotation * glam::mat4(
            glam::vec4(radians.0.cos(), radians.0.sin(), 0.0, 0.0),
            glam::vec4(-radians.0.sin(), radians.0.cos(), 0.0, 0.0),
            glam::vec4(0.0, 0.0, 1.0, 0.0),
            glam::vec4(0.0, 0.0, 0.0, 1.0),
        );
        self
    }

    /// Add scaling.
    ///
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    ///
    /// let t = TransformationMatrixBuilder::new()
    ///           .scale(glam::vec3(2.0, 3.0, 4.0))
    ///           .build();
    /// assert_eq!(t * point!(-4, 6, 8).0, point!(-8.0, 18.0, 32.0).0);
    /// ```
    pub fn scale(mut self, scale: glam::Vec3) -> Self {
        self.scale = glam::Mat4::from_scale(scale);
        self
    }

    /// Add translation.
    ///
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    ///
    /// let t = TransformationMatrixBuilder::new()
    ///           .translation(glam::vec3(5.0, -3.0, 2.0))
    ///           .build();
    /// assert_eq!(t * point!(-3, 4, 5).0, point!(2.0, 1.0, 7.0).0);
    /// ```
    pub fn translation(mut self, translation: glam::Vec3) -> Self {
        self.translation = glam::Mat4::from_translation(translation);
        self
    }

    /// Builds a transformation matrix containing all the operations: rotation, scaling and translation done
    /// correctly.
    pub fn build(self) -> glam::Mat4 {
        // self.translation * self.scale * self.rotation
        self.rotation * self.scale * self.translation
        // self.scale * self.rotation * self.translation
        // self.translation * self.rotation * self.scale
    }
}

/// The view-transformation matrix.
///
/// The transformation matrix for the default orientation, where the orientation
/// looks from the origin along the `z` axis in the negative direction, with `up`
/// to the positive `y` direction.
///
/// ```rust
/// use engine::*;
/// use engine::rt::*;
/// use glam;
/// let from = point!();
///// let to = point!(z -1);
/// let to = point!(0, 0, -1);
/// let up = normalised_direction!(0.0, 1.0, 0.0);
/// let vt = view_transform(from, to, up);
/// assert_eq!(vt, glam::Mat4::IDENTITY);
/// ```
///
/// A view transformation matrix looking in the positive `z` direction.
///
/// ```rust
/// use engine::*;
/// use engine::rt::*;
/// use glam;
/// let from = point!();
/// let to = point!(z 1.0);
/// let up = normalised_direction!(0.0, 1.0, 0.0);
/// let vt = view_transform(from, to, up);
/// assert_eq!(vt, glam::Mat4::from_scale(glam::vec3(-1.0, 1.0, -1.0)));
///// let (scale, _, _) = vt.to_scale_rotation_translation();
///// assert_eq!(scale, glam::vec3(-1.0, 1.0, -1.0));
/// ```
///
/// The view transformation moves the world.
///
/// ```rust
/// use engine::*;
/// use engine::rt::*;
/// use glam;
/// let from = point!(z 8);
/// let to = point!(z 1);
/// let up = normalised_direction!(0.0, 1.0, 0.0);
/// let vt = view_transform(from, to, up);
///// assert_eq!(vt, glam::Mat4::from_translation(glam::vec3(0.0, 0.0, -8.0)));
/// let (_, _, translation) = vt.to_scale_rotation_translation();
/// assert_eq!(translation, glam::vec3(0.0, 0.0, -8.0));
/// ```
///
/// An arbitrary view transformation.
///
/// ```rust
/// use engine::*;
/// use engine::rt::*;
/// use glam;
/// let from = point!(1, 3, 2);
/// let to = point!(4, -2, 8);
/// let up = normalised_direction!(1, 1, 0);
/// let vt = view_transform(from, to, up);
/// assert_eq!(vt, glam::mat4(
///     glam::vec4(-0.50709254, 0.76771593, -0.35856858, 0.0),
///     glam::vec4(0.50709254, 0.6060915, 0.5976143, 0.0),
///     glam::vec4(0.6761234, 0.121218294, -0.71713716, 0.0),
///     glam::vec4(-2.366432, -2.828427, 0.0, 1.0),
/// ));
/// ```
pub fn view_transform(from: Point, to: Point, up: NormalisedDirection) -> glam::Mat4 {
    // An alternative:
    // glam::Mat4::look_at_rh(from.0.xyz(), to.0.xyz(), up.0.xyz())

    // And this is by the book:
    let forward = NormalisedDirection::from(to.0 - from.0);
    let left = forward.xyz().cross(up.xyz());
    let true_up = left.cross(forward.xyz());
    let orientation = glam::mat4(
        glam::vec4(left.x, true_up.x, -forward.x, 0.0),
        glam::vec4(left.y, true_up.y, -forward.y, 0.0),
        glam::vec4(left.z, true_up.z, -forward.z, 0.0),
        glam::vec4(0.0, 0.0, 0.0, 1.0),
    );

    orientation * glam::Mat4::from_translation(-from.0.xyz())
}

/// Returns a number representing an angle in radians calculated by PI division.
///
/// ```rust
/// use engine::*;
/// use engine::rt::*;
///
/// assert_eq!(angle_from_pi_division(2.0).0, 1.5707964);
/// assert_eq!(degree!(90).0, 1.5707964);
/// ```
pub fn angle_from_pi_division(divider: f32) -> Radians {
    Radians(std::f32::consts::PI / divider)
}

/// Represents an angle in degrees.
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Degree(pub f32);
impl From<f32> for Degree {
    fn from(value: f32) -> Self {
        Self(value)
    }
}
impl From<Radians> for Degree {
    fn from(radians: Radians) -> Self {
        Self((radians.0 * 180.0) / std::f32::consts::PI)
    }
}

/// Represents an angle in radians.
///
/// ```rust
/// use engine::*;
/// use engine::rt::*;
/// assert_eq!(angle_from_pi_division(2.0), Radians(1.5707964));
/// assert_eq!(Radians(1.5707964), degree!(90));
/// assert_eq!(Degree::from(Radians(1.5707964)), Degree::from(90.0));
/// ```
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Radians(pub f32);
impl From<f32> for Radians {
    fn from(value: f32) -> Self {
        Self(value)
    }
}
impl From<Degree> for Radians {
    fn from(angle: Degree) -> Self {
        Self((angle.0 * std::f32::consts::PI) / 180.0)
    }
}

/// Represents a camera.
#[derive(Builder, Debug, Default, Copy, Clone, PartialEq)]
pub struct Camera {
    /// Width in pixels (horizontal size).
    pub pixel_width: usize,
    /// Height in pixels (vertical size).
    pub pixel_height: usize,
    /// Field of view (angle).
    pub field_of_view: Radians,
    /// Transform matrix.
    pub transform: glam::Mat4,
}
impl Camera {
    /// Creates new camera.
    pub fn new(
        pixel_width: usize,
        pixel_height: usize,
        field_of_view: Radians,
        transform: glam::Mat4,
    ) -> Self {
        Self {
            pixel_width,
            pixel_height,
            field_of_view,
            transform,
        }
    }

    fn create_ray_with_render_info(&self, x: usize, y: usize, info: &RenderInfo) -> Ray {
        let x_offset = (x as f32 + 0.5) * self.pixel_size();
        let y_offset = (y as f32 + 0.5) * self.pixel_size();
        let world_x = info.half_width - x_offset;
        let world_y = info.half_height - y_offset;
        // let pixel = self.transform.inverse() * point!(world_x, world_y, -1.0).0;
        let pixel = self
            .transform
            .inverse()
            .transform_point3(point!(world_x, world_y, -1.0).0.xyz())
            .extend(1.0);
        // let origin = Point(self.transform.inverse() * point!().0);
        let origin = Point(
            self.transform
                .inverse()
                .transform_point3(point!().0.xyz())
                .extend(1.0),
        );
        let direction = NormalisedDirection::from(pixel - origin.0);
        Ray {
            origin,
            direction,
        }
    }
}

/// Some precomputed basic information about a camera useful for rendering.
#[derive(Builder, Debug, Copy, Clone, PartialEq)]
pub struct RenderInfo {
    /// Half of a field of view.
    pub half_view: f32,
    /// Aspect ratio is a width / height.
    pub aspect_ratio: f32,
    /// Half of the width.
    pub half_width: f32,
    /// Half of the height.
    pub half_height: f32,
}
impl RenderInfo {
    /// Returns a pixel width.
    pub fn pixel_size(&self, pixel_width: usize) -> f32 {
        (self.half_width * 2.0) / pixel_width as f32
    }
}

/// All the camera-related characteristics.
pub trait CameraLike {
    /// Returns width in pixels.
    fn pixel_width(&self) -> usize;
    /// Returns height in pixels.
    fn pixel_height(&self) -> usize;
    /// Returns the field of view.
    fn field_of_view(&self) -> Radians;

    /// Calculates a size of a single pixel.
    ///
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    /// let camera = camera!(200, 125, degree!(90));
    /// assert_eq!(camera.pixel_size(), 0.01);
    /// ```
    fn pixel_size(&self) -> f32 {
        self.render_info().pixel_size(self.pixel_width())
    }

    /// Returns a useful information for rendering.
    fn render_info(&self) -> RenderInfo {
        let half_view = (self.field_of_view().0 / 2.0).tan();
        let aspect_ratio = self.pixel_width() as f32 / self.pixel_height() as f32;
        let half_width;
        let half_height;
        if aspect_ratio >= 1.0 {
            half_width = half_view;
            half_height = half_view / aspect_ratio;
        } else {
            half_width = half_view * aspect_ratio;
            half_height = half_view;
        }
        RenderInfo {
            half_view,
            aspect_ratio,
            half_width,
            half_height,
        }
    }

    /// Creates a ray to go through the specified pixel starting from the camera's origin.
    ///
    /// Constructing a ray through the center of the canvas.
    ///
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    ///
    /// let camera = camera!(201, 101, degree!(90));
    /// let ray = camera.create_ray(100, 50);
    /// assert_eq!(*ray.get_origin(), point!());
    /// assert_eq!(*ray.get_direction(), *normalised_direction!(0.0, 5.9604645e-8, -1.0));
    /// ```
    ///
    /// Constructing a ray through a corner of the canvas.
    ///
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    ///
    /// let camera = camera!(201, 101, degree!(90));
    /// let ray = camera.create_ray(0, 0);
    /// assert_eq!(*ray.get_origin(), point!());
    /// assert_eq!(*ray.get_direction(), *normalised_direction!(0.6651864, 0.33259323, -0.66851234));
    /// ```
    ///
    /// Constructing a ray when the camera is transformed.
    ///
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    ///
    /// let mut camera = camera!(201, 101, degree!(90));
    ///// camera.rotate(glam::vec3(0.0, degree!(45).0, 0.0));
    /// rotate!(camera, y, degree!(45));
    /// translate!(camera, 0, -2, 5);
    /// let ray = camera.create_ray(100, 50);
    /// assert_eq!(*ray.get_direction(), *normalised_direction!(0.7071068, 0.0, -0.7071068));
    /// assert_eq!(*ray.get_origin(), point!(0, 2, -5));
    /// ```
    fn create_ray(&self, x: usize, y: usize) -> Ray;

    /// Renders the world into a bitmap.
    ///
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    /// let world = test_world();
    /// let mut camera = camera!(11, 11, degree!(90));
    /// let from = point!(0, 0, -5);
    /// let to = point!();
    /// let up = normalised_direction!(0, 1, 0);
    ///// *camera.get_transform_matrix_mut() = view_transform(from, to, up);
    /// transform!(camera, view, from, to, up);
    /// let world_like = &world as &dyn WorldLike<_, _>;
    /// let data = camera.render(world_like);
    /// assert_eq!(data[5][5], color!(0.3806612, 0.47582647, 0.2854959));
    /// ```
    fn render<'a, 'b, L: LightLike, O: WorldObject>(
        &self,
        world: &'a (dyn WorldLike<L, O> + 'b),
    ) -> Vec<Vec<ColorRGBA>>
    where
        Ray: ColoredHit<'a, dyn WorldLike<L, O> + 'b>,
    {
        let width = self.pixel_width();
        let height = self.pixel_height();
        let mut data = Vec::with_capacity(width);
        for y in 0..height {
            let mut xdata = Vec::with_capacity(height);
            for x in 0..width {
                let ray = self.create_ray(x, y);
                let color = ray.colored_hit(world);
                xdata.push(color);
            }
            data.push(xdata);
        }
        data
    }
}

impl CameraLike for Camera {
    fn pixel_width(&self) -> usize {
        self.pixel_width
    }

    fn pixel_height(&self) -> usize {
        self.pixel_height
    }

    fn field_of_view(&self) -> Radians {
        self.field_of_view
    }

    fn create_ray(&self, x: usize, y: usize) -> Ray {
        let info = self.render_info();
        self.create_ray_with_render_info(x, y, &info)
    }
}

impl TransformMatrix for Camera {
    fn get_transform_matrix(&self) -> &glam::Mat4 {
        &self.transform
    }

    fn get_transform_matrix_mut(&mut self) -> &mut glam::Mat4 {
        &mut self.transform
    }
}

/// Encapsulates a point.
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Point(pub glam::Vec4);
impl Scale for Point {
    fn scale(&mut self, scale: glam::Vec3) {
        self.0 = glam::Mat4::from_scale(scale)
            .transform_point3(self.0.xyz())
            .extend(self.0.w);
    }
}
impl Translate for Point {
    fn translate(&mut self, translate: glam::Vec3) {
        self.0 += translate.extend(0.0);
    }
}
impl From<glam::Vec4> for Point {
    fn from(value: glam::Vec4) -> Self {
        Self(value)
    }
}
impl From<glam::Vec3> for Point {
    fn from(value: glam::Vec3) -> Self {
        value.extend(1.0).into()
    }
}
impl AsRef<glam::Vec4> for Point {
    fn as_ref(&self) -> &glam::Vec4 {
        &self.0
    }
}
impl AsMut<glam::Vec4> for Point {
    fn as_mut(&mut self) -> &mut glam::Vec4 {
        &mut self.0
    }
}
impl std::borrow::Borrow<glam::Vec4> for Point {
    fn borrow(&self) -> &glam::Vec4 {
        &self.0
    }
}
impl std::borrow::BorrowMut<glam::Vec4> for Point {
    fn borrow_mut(&mut self) -> &mut glam::Vec4 {
        &mut self.0
    }
}
impl Point {
    /// Creates a point.
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        glam::vec3(x, y, z).into()
    }
}

/// Encapsulates an always normalised direction.
/// There are two differences between the [`Direction`] and [`NormalisedDirection`]:
/// 1. The normalised direction always has the `w` parameter set to `0` so that it can't be translated.
/// 2. It is always normalised and of unit length.
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct NormalisedDirection(pub Direction);
impl From<glam::Vec4> for NormalisedDirection {
    fn from(value: glam::Vec4) -> Self {
        Self(Direction(value.xyz().normalize().extend(0.0)))
    }
}
impl From<Direction> for NormalisedDirection {
    fn from(d: Direction) -> Self {
        Self::from(d.0)
    }
}
impl std::ops::Deref for NormalisedDirection {
    type Target = Direction;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for NormalisedDirection {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Encapsulates a direction.
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Direction(glam::Vec4);
impl From<glam::Vec4> for Direction {
    fn from(value: glam::Vec4) -> Self {
        Self(value)
    }
}
impl From<glam::Vec3> for Direction {
    fn from(value: glam::Vec3) -> Self {
        value.extend(1.0).into()
    }
}
impl AsRef<glam::Vec4> for Direction {
    fn as_ref(&self) -> &glam::Vec4 {
        &self.0
    }
}
impl AsMut<glam::Vec4> for Direction {
    fn as_mut(&mut self) -> &mut glam::Vec4 {
        &mut self.0
    }
}
impl Reflect for Direction {
    fn reflect(self, normal: Self) -> Self {
        Self(self.0.reflect(normal.0))
    }
}
impl std::borrow::Borrow<glam::Vec4> for Direction {
    fn borrow(&self) -> &glam::Vec4 {
        &self.0
    }
}
impl std::borrow::BorrowMut<glam::Vec4> for Direction {
    fn borrow_mut(&mut self) -> &mut glam::Vec4 {
        &mut self.0
    }
}
impl std::ops::Deref for Direction {
    type Target = glam::Vec4;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for Direction {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// A sphere.
#[derive(Builder, Debug, Default, Copy, Clone, PartialEq)]
pub struct Sphere {
    /// The radius of the sphere.
    pub radius: f32,
    /// The origin (position, center) of the sphere.
    pub origin: Point,
    /// Transformation matrix.
    pub transform: glam::Mat4,
}

impl Sphere {
    /// Creates a sphere for tests. #[cfg(doctest)] doesn't work.
    pub fn test_sphere() -> Self {
        Self {
            radius: 1.0,
            origin: point!(),
            transform: glam::Mat4::IDENTITY,
        }
    }
}

/// A sphere with material.
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct MaterialSphere<T> {
    /// A sphere.
    sphere: Sphere,
    /// Material of the sphere.
    material: T,
}
impl<T> MaterialSphere<T> {
    /// Create a material sphere.
    pub fn new(sphere: Sphere, material: T) -> Self {
        Self { sphere, material }
    }

    /// Return a reference to the sphere.
    pub fn get_sphere(&self) -> &Sphere {
        &self.sphere
    }

    /// Return a mutable reference to the sphere.
    pub fn get_sphere_mut(&mut self) -> &mut Sphere {
        &mut self.sphere
    }

    /// Set a new sphere.
    pub fn set_sphere(&mut self, sphere: Sphere) {
        self.sphere = sphere;
    }
}

/// A ray.
#[derive(Builder, Debug, Default, Copy, Clone, PartialEq)]
pub struct Ray {
    /// The origin (position) of the ray.
    origin: Point,
    /// A direction vector.
    #[builder(setter(custom))]
    direction: NormalisedDirection,
}
impl Ray {
    /// Creates a new ray with normalised direction vector.
    pub fn new(origin: Point, direction: NormalisedDirection) -> Self {
        Self { origin, direction }
    }

    /// Returns a position of a vector if it was advanced in its direction by [`value`].
    pub fn advanced_position(&self, value: f32) -> Point {
        (self.origin.0 + self.direction.0 .0 * value).into()
    }
}
impl RayBuilder {
    /// Sets the direction for a vector.
    pub fn direction(&mut self, direction: NormalisedDirection) -> &mut Self {
        self.direction = Some(direction);
        self
    }
}

/// A world is a collection of various objects.
#[derive(Debug, Default, Clone)]
pub struct World<L: LightLike, O: WorldObject> {
    /// All the light sources in the world.
    pub light_sources: Vec<L>,
    /// All the objects in the world.
    pub objects: Vec<Box<O>>,
}

impl<L: LightLike, O: WorldObject> World<L, O> {
    /// Return a new world containing light sources and ray-hittable objects.
    pub fn new(light_sources: Vec<L>, objects: Vec<Box<O>>) -> Self {
        Self {
            light_sources,
            objects,
        }
    }
}

/// Return a funny test world.
pub fn test_world() -> World<PointLight, MaterialSphere<PhongMaterial>> {
    let light_source = PointLightBuilder::default()
        .origin(point!(-10.0, 10.0, -10.0))
        .color(ColorRGBA::white())
        .build()
        .unwrap();
    // Page 161.
    let material = PhongMaterial::new(0.1, 0.7, 0.2, 200.0, glam::vec4(0.8, 1.0, 0.6, 1.0));
    let sphere1 = Sphere::test_sphere();
    let sphere1 = Box::new(MaterialSphere::new(sphere1, material));
    let mut sphere2 = Sphere::test_sphere();
    sphere2.scale(glam::vec3(0.5, 0.5, 0.5));
    let material = PhongMaterial::test_material();
    let sphere2 = Box::new(MaterialSphere::new(sphere2, material));
    World::new(vec![light_source], vec![sphere1, sphere2])
}

/// Return a funny test world number 2.
/// Page 195.
pub fn test_world_2() -> World<PointLight, MaterialSphere<PhongMaterial>> {
    let mut floor = Sphere::test_sphere();
    floor.scale(glam::vec3(10.0, 0.01, 10.0));
    let mut material = PhongMaterial::test_material();
    material.color = color!(1, 0.9, 0.9);
    material.specular = 0.0;
    let floor = Box::new(MaterialSphere::new(floor, material));

    let mut left_wall = Sphere::test_sphere();
    left_wall.scale(glam::vec3(10.0, 0.01, 10.0));
    left_wall.rotate_x(degree!(90));
    left_wall.rotate_y(degree!(-45));
    left_wall.translate(glam::vec3(0.0, 0.0, 5.0));
    // transform!(left_wall, TransformationMatrixBuilder::new()
    //            .scale(glam::vec3(10.0, 0.01, 10.0))
    //            .rotation([degree!(90), degree!(-45), degree!(0.0)])
    //            .translation(glam::vec3(0.0, 0.0, 5.0))
    //            .build());
    // let t = TransformationMatrixBuilder::new().translation(glam::vec3(0.0, 0.0, 5.0)).build();
    // let ry = TransformationMatrixBuilder::new().rotate_y(degree!(-45)).build();
    // let rx = TransformationMatrixBuilder::new().rotate_x(degree!(90)).build();
    // let s = TransformationMatrixBuilder::new().scale(glam::vec3(10.0, 0.01, 10.0)).build();
    // left_wall.transform = s * ry * rx * t;
    // left_wall.transform = t * ry * rx * s;
    let left_wall = Box::new(MaterialSphere::new(left_wall, material));

    let mut right_wall = Sphere::test_sphere();
    right_wall.scale(glam::vec3(10.0, 0.01, 10.0));
    // right_wall.scale(glam::vec3(3.0, 2.0, 3.0));
    right_wall.rotate_x(degree!(90));
    right_wall.rotate_y(degree!(45));
    right_wall.translate(glam::vec3(0.0, 0.0, 5.0));
    // transform!(right_wall, TransformationMatrixBuilder::new()
    //            .scale(glam::vec3(10.0, 0.01, 10.0))
    //            .rotation([degree!(90), degree!(45), degree!(0.0)])
    //            .translation(glam::vec3(0.0, 0.0, 5.0))
    //            .build());
    let right_wall = Box::new(MaterialSphere::new(right_wall, material));

    let mut middle = Sphere::test_sphere();
    middle.translate(glam::vec3(-0.5, 1.0, 0.5));
    let mut material = PhongMaterial::test_material();
    material.color = color!(0.1, 1, 0.5);
    material.diffuse = 0.7;
    material.specular = 0.3;
    let middle = Box::new(MaterialSphere::new(middle, material));

    let mut right = Sphere::test_sphere();
    right.translate(glam::vec3(1.5, 0.5, -0.5));
    right.scale(glam::vec3(0.5, 0.5, 0.5));
    let mut material = PhongMaterial::test_material();
    material.color = color!(0.5, 1, 0.1);
    material.diffuse = 0.7;
    material.specular = 0.3;
    let right = Box::new(MaterialSphere::new(right, material));

    let mut left = Sphere::test_sphere();
    left.translate(glam::vec3(-1.5, 0.33, -0.75));
    left.scale(glam::vec3(0.33, 0.33, 0.33));
    let mut material = PhongMaterial::test_material();
    material.color = color!(1, 0.8, 0.1);
    material.diffuse = 0.7;
    material.specular = 0.3;
    let left = Box::new(MaterialSphere::new(left, material));

    let light_source = PointLightBuilder::default()
        .origin(point!(-10.0, 10.0, -10.0))
        .color(ColorRGBA::white())
        .build()
        .unwrap();
    World::new(
        vec![light_source],
        vec![floor, left_wall, right_wall, middle, right, left],
    )
}

/// Describes a world-like object.
pub trait WorldLike<L, O: std::fmt::Debug> {
    /// Return all the light sources in the world.
    fn get_light_sources(&self) -> &[L];
    /// Return mutable references to the light sources in the world.
    fn get_light_sources_mut(&mut self) -> &mut [L];
    /// Return all the ray-hittable objects in the world.
    fn get_objects(&self) -> Vec<&O>;
    /// Return mutable references to the objects in the world.
    fn get_objects_mut(&mut self) -> Vec<&mut O>;
    /// Returns a color that a camera should see with these computations and world.
    fn shade_hit<'a>(&'a self, computations: &PreparedComputations<'a, O>) -> ColorRGBA;
}

impl<L: LightLike, O: WorldObject + HasMaterial<PhongMaterial>> WorldLike<L, O> for World<L, O> {
    fn get_light_sources(&self) -> &[L] {
        self.light_sources.as_slice()
    }

    fn get_light_sources_mut(&mut self) -> &mut [L] {
        self.light_sources.as_mut_slice()
    }

    fn get_objects(&self) -> Vec<&O> {
        self.objects.iter().map(|o| o.as_ref()).collect()
    }

    fn get_objects_mut(&mut self) -> Vec<&mut O> {
        self.objects.iter_mut().map(|o| o.as_mut()).collect()
    }

    /// Shading an intersection.
    ///
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    /// let world = test_world();
    /// let ray = RayBuilder::default()
    ///     .origin(point!(0.0, 0.0, -5.0))
    ///     .direction(normalised_direction!(0.0, 0.0, 1.0))
    ///     .build().unwrap();
    /// let world_like = &world as &dyn WorldLike<_, _>;
    /// let intersection = Intersection { object: world.objects[0].as_ref(), value: 4.0f32 };
    /// let pc = PreparedComputations::new(intersection, ray);
    /// assert_eq!(world.shade_hit(&pc), glam::vec4(0.3806612, 0.47582647, 0.2854959, 1.0).into());
    /// ```
    ///
    /// Shading an intersection from the inside.
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    /// let mut world = test_world();
    /// world.light_sources[0].origin = point!(0.0, 0.25, 0.0);
    /// world.light_sources[0].color = ColorRGBA::white();
    /// let ray = RayBuilder::default()
    ///     .origin(point!())
    ///     .direction(normalised_direction!(0.0, 0.0, 1.0))
    ///     .build().unwrap();
    /// let world_like = &world as &dyn WorldLike<_, _>;
    /// let intersection = Intersection { object: world.objects[1].as_ref(), value: 0.5f32 };
    /// let pc = PreparedComputations::new(intersection, ray);
    /// assert_eq!(world.shade_hit(&pc), glam::vec4(0.9049845, 0.9049845, 0.9049845, 1.0).into());
    /// ```
    fn shade_hit<'a>(&'a self, computations: &PreparedComputations<'a, O>) -> ColorRGBA {
        self.light_sources
            .iter()
            .fold(ColorRGBA::black(), |accum, light| {
                ColorRGBA(
                    (accum.0.xyz()
                        + light
                            .light_at_phong_at_point(
                                computations.intersection.object,
                                computations.point,
                                computations.eye_direction,
                                computations.normal,
                            )
                            .0
                            .xyz())
                    .extend(accum.0.w),
                )
            })
    }
}

/// Encapsulates a color in the RGBA format.
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct ColorRGBA(pub glam::Vec4);
impl From<glam::Vec4> for ColorRGBA {
    fn from(vec4: glam::Vec4) -> Self {
        Self(vec4)
    }
}
impl ColorRGBA {
    /// Return black color.
    pub fn black() -> Self {
        glam::vec4(0.0, 0.0, 0.0, 1.0).into()
    }

    /// Return white color.
    pub fn white() -> Self {
        glam::vec4(1.0, 1.0, 1.0, 1.0).into()
    }
}
/// Indicates that the implementor has colors.
pub trait HasColorRGBA {
    /// Return the red component.
    fn r(&self) -> f32;

    /// Return the green component.
    fn g(&self) -> f32;

    /// Return the blue component.
    fn b(&self) -> f32;

    /// Return the alpha component.
    fn a(&self) -> f32;
}

impl HasColorRGBA for ColorRGBA {
    fn r(&self) -> f32 {
        self.0.x
    }

    fn g(&self) -> f32 {
        self.0.y
    }

    fn b(&self) -> f32 {
        self.0.z
    }

    fn a(&self) -> f32 {
        self.0.w
    }
}

/// Describes a light-like object.
pub trait LightLike {
    /// Set the light at the object. Returns the color with which a pixel should be coloured with.
    ///
    /// Lighting with the eye between light and surface.
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    /// let sphere = Sphere::test_sphere();
    /// let material = PhongMaterial::new(0.1, 0.9, 0.9, 200.0, ColorRGBA::white());
    /// let sphere = MaterialSphere::new(sphere, material);
    /// let eye = normalised_direction!(0.0, 0.0, -1.0);
    /// let normal = normalised_direction!(0.0, 0.0, -1.0);
    /// let light = PointLightBuilder::default()
    ///     .origin(point!(0.0, 0.0, -10.0))
    ///     .color(color!(1.0, 1.0, 1.0, 1.0))
    ///     .build().unwrap();
    /// let pixel_color = light.light_at_phong(&sphere, eye, normal);
    /// assert_eq!(pixel_color.0, glam::vec4(1.9, 1.9, 1.9, 1.9));
    /// ```
    ///
    /// Lighting with the eye between light and surface, eye offset 45*.
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    /// let sphere = Sphere::test_sphere();
    /// let material = PhongMaterial::new(0.1, 0.9, 0.9, 200.0, ColorRGBA::white());
    /// let sphere = MaterialSphere::new(sphere, material);
    /// let eye = normalised_direction!(0.0, 2.0f32.sqrt() / 2.0, -2.0f32.sqrt() / 2.0);
    /// let normal = normalised_direction!(0.0, 0.0, -1.0);
    /// let light = PointLightBuilder::default()
    ///     .origin(point!(0.0, 0.0, -10.0))
    ///     .color(color!(1.0, 1.0, 1.0, 1.0))
    ///     .build().unwrap();
    /// let pixel_color = light.light_at_phong(&sphere, eye, normal);
    /// assert_eq!(pixel_color.0, glam::vec4(1.0, 1.0, 1.0, 1.0));
    /// ```
    ///
    /// Lighting with the eye opposite surface, light offset 45*.
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    /// let sphere = Sphere::test_sphere();
    /// let material = PhongMaterial::new(0.1, 0.9, 0.9, 200.0, ColorRGBA::white());
    /// let sphere = MaterialSphere::new(sphere, material);
    /// let eye = normalised_direction!(0.0, 0.0, -1.0);
    /// let normal = normalised_direction!(0.0, 0.0, -1.0);
    /// let light = PointLightBuilder::default()
    ///     .origin(point!(0.0, 10.0, -10.0))
    ///     .color(color!(1.0, 1.0, 1.0, 1.0))
    ///     .build().unwrap();
    /// let pixel_color = light.light_at_phong(&sphere, eye, normal);
    ///// let expected_color = material.ambient + material.diffuse * (2.0f32.sqrt() / 2.0) + 0.0f32;
    ///// assert_eq!(pixel_color.0, color!(expected_color).0);
    /// assert_eq!(pixel_color.0, color!(0.73639613).0);
    /// ```
    ///
    /// Lighting with the eye in the path of the reflection vector.
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    /// let sphere = Sphere::test_sphere();
    /// let material = PhongMaterial::new(0.1, 0.9, 0.9, 200.0, ColorRGBA::white());
    /// let sphere = MaterialSphere::new(sphere, material);
    /// let eye = normalised_direction!(0.0, -2.0f32.sqrt() / 2.0, -2.0f32.sqrt() / 2.0);
    /// let normal = normalised_direction!(0.0, 0.0, -1.0);
    /// let light = PointLightBuilder::default()
    ///     .origin(point!(0.0, 10.0, -10.0))
    ///     .color(color!(1.0, 1.0, 1.0, 1.0))
    ///     .build().unwrap();
    /// let pixel_color = light.light_at_phong(&sphere, eye, normal);
    ///// let expected_color = material.ambient + material.diffuse * (2.0f32.sqrt() / 2.0f32) + material.specular;
    ///// assert_eq!(pixel_color.0, color!(expected_color).0);
    /// assert_eq!(pixel_color.0, color!(1.6364176).0);
    /// ```
    ///
    /// Lighting with the light behind the surface.
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    /// let sphere = Sphere::test_sphere();
    /// let material = PhongMaterial::new(0.1, 0.9, 0.9, 200.0, ColorRGBA::white());
    /// let sphere = MaterialSphere::new(sphere, material);
    /// let eye = normalised_direction!(0.0, 0.0, -1.0);
    /// let normal = normalised_direction!(0.0, 0.0, -1.0);
    /// let light = PointLightBuilder::default()
    ///     .origin(point!(0.0, 0.0, 10.0))
    ///     .color(color!(1.0, 1.0, 1.0, 1.0))
    ///     .build().unwrap();
    /// let pixel_color = light.light_at_phong(&sphere, eye, normal);
    /// assert_eq!(pixel_color.0, color!(0.1, 0.1, 0.1, 2.1).0);
    /// ```
    fn light_at_phong(
        &self,
        object: &(dyn PhongLit),
        camera_direction: NormalisedDirection,
        normal: NormalisedDirection,
    ) -> ColorRGBA;

    /// Same as above but instead of object's origin uses a specialised point on it.
    fn light_at_phong_at_point(
        &self,
        object: &(dyn PhongLit),
        point: Point,
        camera_direction: NormalisedDirection,
        normal: NormalisedDirection,
    ) -> ColorRGBA;
}

/// Indicates that an object can be phong-lit.
pub trait PhongLit: HasOrigin + HasMaterial<PhongMaterial> {}
impl<T> PhongLit for T where T: HasOrigin + HasMaterial<PhongMaterial> {}

/// A point light.
#[derive(Builder, Debug, Default, Copy, Clone, PartialEq)]
pub struct PointLight {
    /// The origin of the point light.
    pub origin: Point,
    /// Color of the light source, xyzw == rgba.
    pub color: ColorRGBA,
}
impl LightLike for PointLight {
    fn light_at_phong(
        &self,
        object: &(dyn PhongLit),
        camera_direction: NormalisedDirection,
        normal: NormalisedDirection,
    ) -> ColorRGBA {
        self.light_at_phong_at_point(object, *object.get_origin(), camera_direction, normal)
    }

    fn light_at_phong_at_point(
        &self,
        object: &(dyn PhongLit),
        point: Point,
        camera_direction: NormalisedDirection,
        normal: NormalisedDirection,
    ) -> ColorRGBA {
        let material = object.get_material();
        let effective_color = ColorRGBA(material.color.0 * self.color.0.w);
        let light_direction = NormalisedDirection::from(self.origin.0 - point.0);
        let ambient = ColorRGBA(effective_color.0 * material.ambient);
        let light_dot_normal = light_direction.dot(**normal);
        let diffuse;
        let specular;
        if light_dot_normal < 0.0 {
            diffuse = ColorRGBA::black();
            specular = ColorRGBA::black();
        } else {
            diffuse = ColorRGBA(effective_color.0 * material.diffuse * light_dot_normal);
            let reflect_direction = (-light_direction.0 .0).reflect(**normal);
            let reflect_dot_eye = reflect_direction.dot(**camera_direction);
            if reflect_dot_eye < 0.0 {
                specular = ColorRGBA::black();
            } else {
                let factor = reflect_dot_eye.powf(material.shininess);
                specular = ColorRGBA(self.color.0 * material.specular * factor);
            }
        }
        (ambient.0 + diffuse.0 + specular.0).into()
    }
}

/// Defines a material using [`glam::Vec4`] as a storage.
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct PhongMaterial {
    /// The material's ambient property.
    pub ambient: f32,
    /// The material's diffuse property.
    pub diffuse: f32,
    /// The material's specular property.
    pub specular: f32,
    /// The material's shininess property.
    pub shininess: f32,
    /// The color of the object with this material.
    pub color: ColorRGBA,
}

impl PhongMaterial {
    /// Create a new phong material.
    pub fn new<T: Into<ColorRGBA>>(
        ambient: f32,
        diffuse: f32,
        specular: f32,
        shininess: f32,
        color: T,
    ) -> Self {
        Self {
            ambient,
            diffuse,
            specular,
            shininess,
            color: color.into(),
        }
    }

    /// Page 161.
    /// Ambient = 0.1, diffuse = 0.9, specular = 0.9, shininess = 200.0, color = white.
    fn test_material() -> Self {
        Self {
            ambient: 0.1,
            diffuse: 0.9,
            specular: 0.9,
            shininess: 200.0,
            color: ColorRGBA::white(),
        }
    }
}

impl From<glam::Vec4> for PhongMaterial {
    fn from(vec4: glam::Vec4) -> Self {
        Self {
            ambient: vec4.x,
            diffuse: vec4.y,
            specular: vec4.z,
            shininess: vec4.w,
            color: ColorRGBA::black(),
        }
    }
}

/// Assigns a material.
pub trait AssignMaterial<T> {
    /// Assigns a material.
    fn assign_material(&mut self, material: T);
}

impl<T> AssignMaterial<T> for MaterialSphere<T> {
    fn assign_material(&mut self, material: T) {
        self.material = material;
    }
}

/// Indicates that Self has a material.
pub trait HasMaterial<T> {
    /// Return a reference to the material.
    fn get_material(&self) -> &T;

    /// Return a mutable reference to the material.
    fn get_material_mut(&mut self) -> &mut T;

    /// Set a material.
    fn set_material(&mut self, material: T);
}

impl<T> HasMaterial<T> for MaterialSphere<T> {
    /// Return a reference to the material.
    fn get_material(&self) -> &T {
        &self.material
    }

    /// Return a mutable reference to the material.
    fn get_material_mut(&mut self) -> &mut T {
        &mut self.material
    }

    /// Set a material.
    fn set_material(&mut self, material: T) {
        self.material = material;
    }
}

/// All the traits of a spherical shape.
pub trait Spherical {
    /// Get the radius of the sphere.
    fn get_radius(&self) -> f32;
}

impl Spherical for Sphere {
    fn get_radius(&self) -> f32 {
        self.radius
    }
}

impl<T> Spherical for MaterialSphere<T> {
    fn get_radius(&self) -> f32 {
        self.sphere.get_radius()
    }
}
/// Indicates that an implementor is like a sphere.
pub trait SphereLike: Spherical + HasOrigin + TransformMatrix {}
impl<T> SphereLike for T where T: Spherical + HasOrigin + TransformMatrix {}

impl<T> TransformMatrix for MaterialSphere<T> {
    fn get_transform_matrix(&self) -> &glam::Mat4 {
        self.sphere.get_transform_matrix()
    }

    fn get_transform_matrix_mut(&mut self) -> &mut glam::Mat4 {
        self.sphere.get_transform_matrix_mut()
    }
}

/// Contains some prepared computations useful for reflection and refraction.
#[derive(Copy, Clone, Debug)]
pub struct PreparedComputations<'a, T: std::fmt::Debug + ?Sized> {
    /// The intersection.
    pub intersection: Intersection<'a, T>,
    /// The position of the ray intersecting the object.
    pub point: Point,
    /// Direction to the camera.
    pub eye_direction: NormalisedDirection,
    /// Intersected's object normal at [`point`].
    pub normal: NormalisedDirection,
    /// If the ray originated from within the object (from the inside).
    pub is_inside: bool,
}

impl<'a, T> PreparedComputations<'a, T>
where
    T: NormalAt + std::fmt::Debug,
{
    /// Create a new prepared computation.
    ///
    /// The hit, when an intersection occurs on the inside.
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    /// let sphere = Sphere::test_sphere();
    /// let ray = RayBuilder::default()
    ///     .origin(point!(0.0, 0.0, 0.0))
    ///     .direction(normalised_direction!(0.0, 0.0, 1.0))
    ///     .build().unwrap();
    /// let intersection = Intersection { object: &sphere, value: 1.0f32 };
    /// let computation = PreparedComputations::new(intersection, ray);
    /// assert_eq!(computation.point, point!(0.0, 0.0, 1.0));
    /// assert_eq!(computation.eye_direction, normalised_direction!(0.0, 0.0, -1.0));
    /// assert_eq!(computation.is_inside, true);
    /// assert_eq!(computation.normal, normalised_direction!(0.0, 0.0, -1.0));
    /// ```
    ///
    /// The hit, when an intersection occurs on the outside.
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    /// let sphere = Sphere::test_sphere();
    /// let ray = RayBuilder::default()
    ///     .origin(point!(0.0, 0.0, -5.0))
    ///     .direction(normalised_direction!(0.0, 0.0, 1.0))
    ///     .build().unwrap();
    /// let intersection = Intersection { object: &sphere, value: 4.0f32 };
    /// let computation = PreparedComputations::new(intersection, ray);
    /// assert_eq!(computation.is_inside, false);
    /// ```
    ///
    /// Precomputing the state of an intersection.
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    /// let sphere = Sphere::test_sphere();
    /// let ray = RayBuilder::default()
    ///     .origin(point!(0.0, 0.0, -5.0))
    ///     .direction(normalised_direction!(0.0, 0.0, 1.0))
    ///     .build().unwrap();
    /// let intersection = Intersection { object: &sphere, value: 4.0f32 };
    /// let computation = PreparedComputations::new(intersection, ray);
    /// assert_eq!(computation.intersection.object, &sphere);
    /// assert_eq!(computation.intersection.value, 4.0f32);
    /// assert_eq!(computation.point, point!(0.0, 0.0, -1.0));
    /// assert_eq!(computation.eye_direction, normalised_direction!(0.0, 0.0, -1.0));
    /// assert_eq!(computation.normal, normalised_direction!(0.0, 0.0, -1.0));
    /// assert_eq!(computation.is_inside, false);
    /// ```
    ///
    pub fn new(intersection: Intersection<'a, T>, ray: Ray) -> Self {
        let object_point = ray.advanced_position(intersection.value).0;
        let eye_direction = -ray.direction.0 .0;
        let mut normal = intersection.object.normal_at(object_point.into());
        let is_inside = if normal.0.dot(eye_direction) < 0.0 {
            true
        } else {
            false
        };
        if is_inside {
            normal = (-**normal).into();
        }
        Self {
            intersection,
            point: object_point.into(),
            eye_direction: eye_direction.into(),
            normal,
            is_inside,
        }
    }
}

/// Returns the direction towards a particular point in space.
pub trait DirectionTo {
    /// Returns the direction towards a particular point in space.
    fn direction_to(&self, point: Point) -> Direction;
}
impl<T> DirectionTo for T
where
    T: HasOrigin,
{
    fn direction_to(&self, point: Point) -> Direction {
        (self.get_origin().0 - point.0).into()
    }
}
impl DirectionTo for Point {
    fn direction_to(&self, point: Point) -> Direction {
        (self.0 - point.0).into()
    }
}

/// Changes the direction of the view so that it looks at a particular point in space.
pub trait LookAt {
    /// Changes the direction of the view so that it looks at a particular point in space.
    fn look_at(&mut self, point: Point);
}

impl<T> LookAt for T
where
    T: HasDirection + HasOrigin,
{
    fn look_at(&mut self, point: Point) {
        let origin = self.get_origin().clone();
        let direction = self.get_direction_mut();
        *direction = (point.0 - origin.0).into();
    }
}

/// Reflects the direction of a vector over a normal.
pub trait Reflect<T = Self> {
    /// Reflects the direction of a vector.
    ///
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    /// assert_eq!(glam::vec3(1.0, -1.0, 0.0).reflect(glam::vec3(0.0, 1.0, 0.0)), glam::vec3(1.0, 1.0, 0.0));
    /// let num = 2.0f32.sqrt() / 2.0f32;
    //    /// assert_eq!(glam::vec3(0.0, -1.0, 0.0).reflect(glam::vec3(num, num, 0.0)), glam::vec3(1.0, 0.0, 0.0));
    /// let reflected = glam::vec3(0.0, -1.0, 0.0).reflect(glam::vec3(num, num, 0.0));
    /// let expected = glam::vec3(1.0, 0.0, 0.0);
    /// assert!(reflected.x <= expected.x + std::f32::EPSILON);
    /// assert!(reflected.x >= expected.x - std::f32::EPSILON);
    /// assert!(reflected.y <= expected.y + std::f32::EPSILON);
    /// assert!(reflected.y >= expected.y - std::f32::EPSILON);
    /// assert!(reflected.z <= expected.z + std::f32::EPSILON);
    /// assert!(reflected.z >= expected.z - std::f32::EPSILON);
    /// ```
    fn reflect(self, normal: Self) -> Self;
}

impl Reflect for glam::Vec3 {
    fn reflect(self, normal: Self) -> Self {
        let reflected = self.clone();
        reflected - normal * 2.0f32 * reflected.dot(normal)
    }
}

impl Reflect for glam::Vec4 {
    fn reflect(self, normal: Self) -> Self {
        self.xyz().reflect(normal.xyz()).extend(1.0)
    }
}

/// Calculates the surface normal vector of an object at a certain point.
pub trait NormalAt
where
    Self: HasOrigin + TransformMatrix,
{
    /// Calculates the surface normal vector of an object at a certain point.
    /// Example with a sphere at origin (0;0;0)
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    /// let mut sphere = Sphere::test_sphere();
    /// assert_eq!(sphere.normal_at(point!(0.577, 0.577, 0.577)), normalised_direction!(0.5773503, 0.5773503, 0.5773503));
    /// assert_eq!(sphere.normal_at(point!(1.0, 0.0, 0.0)), normalised_direction!(1.0, 0.0, 0.0));
    /// translate!(sphere, y, 1.0);
    /// assert_eq!(sphere.normal_at(point!(0.0, 1.70711, -0.70711)), normalised_direction!(0.0, 0.7071068, -0.70710677));
    /// translate!(sphere, y, -1.0);
    /// scale!(sphere, y, 0.5);
    ///// sphere.rotate(glam::vec3(0.0, 0.0, std::f32::consts::PI / 5.0f32));
    /// rotate!(sphere, z, angle_from_pi_division(5.0));
    /// let num = 2.0f32.sqrt() / 2.0f32;
    /// assert_eq!(sphere.normal_at(point!(0.0, num, -num)), normalised_direction!(-2.0444226e-8, 0.97014254, -0.24253564));
    /// ```
    fn normal_at(&self, point: Point) -> NormalisedDirection;
}

impl<T> NormalAt for T
where
    T: HasOrigin + TransformMatrix,
{
    fn normal_at(&self, world_point: Point) -> NormalisedDirection {
        let transform_inversed = self.get_transform_matrix().inverse();
        let object_point = transform_inversed * world_point.0;
        let origin = self.get_origin().0;
        let object_normal = object_point - origin;
        (transform_inversed.transpose() * object_normal).into()
    }
}

/// Indicates that an entity can have a transformation through the transformation matrix.
pub trait TransformMatrix {
    /// Returns read-only transformation matrix.
    fn get_transform_matrix(&self) -> &glam::Mat4;
    /// Returns a mutable transformation matrix.
    fn get_transform_matrix_mut(&mut self) -> &mut glam::Mat4;
}

impl TransformMatrix for Sphere {
    fn get_transform_matrix(&self) -> &glam::Mat4 {
        &self.transform
    }

    fn get_transform_matrix_mut(&mut self) -> &mut glam::Mat4 {
        &mut self.transform
    }
}

/// Indicates that an entity has an origin.
pub trait HasOrigin {
    /// Returns the origin.
    fn get_origin(&self) -> &Point;
    /// Returns a mutable reference to the origin.
    fn get_origin_mut(&mut self) -> &mut Point;
}

/// Indicates that an entity has a direction.
pub trait HasDirection {
    /// Returns the direction.
    fn get_direction(&self) -> &Direction;
    /// Returns a mutable reference to the direction.
    fn get_direction_mut(&mut self) -> &mut Direction;
}

impl HasOrigin for Sphere {
    fn get_origin(&self) -> &Point {
        &self.origin
    }

    fn get_origin_mut(&mut self) -> &mut Point {
        &mut self.origin
    }
}

impl<T> HasOrigin for MaterialSphere<T> {
    fn get_origin(&self) -> &Point {
        self.sphere.get_origin()
    }

    fn get_origin_mut(&mut self) -> &mut Point {
        self.sphere.get_origin_mut()
    }
}

impl HasOrigin for Ray {
    fn get_origin(&self) -> &Point {
        &self.origin
    }

    fn get_origin_mut(&mut self) -> &mut Point {
        &mut self.origin
    }
}

impl HasDirection for Ray {
    fn get_direction(&self) -> &Direction {
        &self.direction
    }

    fn get_direction_mut(&mut self) -> &mut Direction {
        &mut self.direction
    }
}

/// Translates an entity (moves it).
pub trait Translate {
    /// Translates an entity (moves it).
    fn translate(&mut self, translation: glam::Vec3);

    /// Translates an entity with a vec4 ignoring the `w` value.
    fn translate_4(&mut self, translate: glam::Vec4) {
        self.translate(translate.xyz())
    }
}
impl<T> Translate for T
where
    T: TransformMatrix,
{
    /// Translates an entity (moves it).
    /// In this example it doesn't move the origin point,
    /// but records the translation into the transform matrix.
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    /// let mut sphere = Sphere::test_sphere();
    /// translate!(sphere, 2, 3, 4);
    /// let transform = glam::mat4(
    ///    glam::vec4(1.0, 0.0, 0.0, 0.0),
    ///    glam::vec4(0.0, 1.0, 0.0, 0.0),
    ///    glam::vec4(0.0, 0.0, 1.0, 0.0),
    ///    glam::vec4(2.0, 3.0, 4.0, 1.0),
    /// );
    /// assert_eq!(sphere.transform, transform);
    /// assert_eq!(sphere.origin, point!(0.0, 0.0, 0.0));
    /// ```
    fn translate(&mut self, translate: glam::Vec3) {
        let transform = self.get_transform_matrix_mut();
        // transform.w_axis.x += translate.x;
        // transform.w_axis.y += translate.y;
        // transform.w_axis.z += translate.z;
        // TODO: Plus works better in math in with angles, multiplication with positioning.
        *transform *= glam::Mat4::from_translation(translate);
    }
}

/// Translates the origin (the point).
pub trait TranslateOrigin
where
    Self: HasOrigin,
{
    /// Translates the origin (the point).
    fn translate_origin(&mut self, translation: glam::Vec3);

    /// Translates the origin (the point) while not taking the `w` parameter into account.
    fn translate_origin_4(&mut self, translation: glam::Vec4) {
        self.translate_origin(translation.xyz())
    }
}

impl<T> TranslateOrigin for T
where
    T: HasOrigin,
{
    fn translate_origin(&mut self, translation: glam::Vec3) {
        let origin = self.get_origin_mut();
        *origin = Point(origin.0 + translation.extend(0.0));
    }
}

/// Shearing (or skew) transformation.
pub trait Sheer {
    /// Shearing (or skew) transformation.
    fn sheer(
        &mut self,
        x_to_y: f32,
        x_to_z: f32,
        y_to_x: f32,
        y_to_z: f32,
        z_to_x: f32,
        z_to_y: f32,
    );
}

impl<T> Sheer for T
where
    T: TransformMatrix,
{
    fn sheer(
        &mut self,
        x_to_y: f32,
        x_to_z: f32,
        y_to_x: f32,
        y_to_z: f32,
        z_to_x: f32,
        z_to_y: f32,
    ) {
        let transform = self.get_transform_matrix_mut();
        let sheer_matrix = glam::mat4(
            glam::vec4(1.0, x_to_y, x_to_z, 0.0),
            glam::vec4(y_to_x, 1.0, y_to_z, 0.0),
            glam::vec4(z_to_x, z_to_y, 1.0, 0.0),
            glam::vec4(0.0, 0.0, 0.0, 1.0),
        );
        *transform *= sheer_matrix;
    }
}

/// Rotates an object around axis.
pub trait Rotate {
    /// Rotates an object around axis.
    ///
    /// Rotating a point around the X axis.
    ///
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    ///
    /// let mut p = point!(0, 1, 0);
    /// p.rotate_x(degree!(45));
    /// assert_eq!(p, point!(0.0, 0.70710677, 0.70710677));
    /// let mut p = point!(0, 1, 0);
    /// p.rotate_x(degree!(90));
    /// assert_eq!(p, point!(0, -4.371139e-8, 1));
    /// ```
    ///
    /// The inverse of an X-rotation rotates in the opposite direction.
    ///
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    ///
    /// let mut p = point!(0, 1, 0);
    /// p.rotate_x(degree!(-45));
    /// assert_eq!(p, point!(0.0, 0.70710677, -0.70710677));
    /// ```
    fn rotate_x(&mut self, radians: Radians);
    /// Same as [`rotate_x`] but by the Y axis.
    ///
    /// Rotating a point around the Y axis.
    ///
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    ///
    /// let mut p = point!(0, 0, 1);
    /// p.rotate_y(degree!(45));
    /// assert_eq!(p, point!(0.70710677, 0.0, 0.70710677));
    /// let mut p = point!(0, 0, 1);
    /// p.rotate_y(degree!(90));
    /// assert_eq!(p, point!(1.0, 0.0, -4.371139e-8));
    /// ```
    fn rotate_y(&mut self, radians: Radians);
    /// Same as [`rotate_x`] but by the Z axis.
    ///
    /// Rotating a point around the Z axis.
    ///
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    ///
    /// let mut p = point!(0, 1, 0);
    /// p.rotate_z(degree!(45));
    /// assert_eq!(p, point!(-0.70710677, 0.70710677, 0.0));
    /// let mut p = point!(0, 1, 0);
    /// p.rotate_z(degree!(90));
    /// assert_eq!(p, point!(-1.0, -4.371139e-8, 0.0));
    /// ```
    fn rotate_z(&mut self, radians: Radians);

    /// Rotates in general way. Three coordinates - x, y and z respective in a three-element array.
    ///
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    ///
    /// let mut p1 = point!(0, 1, 0);
    /// p1.rotate_z(degree!(45));
    /// assert_eq!(p1, point!(-0.70710677, 0.70710677, 0.0));
    /// let mut p2 = point!(0, 1, 0);
    /// p2.rotate([degree!(0), degree!(0), degree!(45)]);
    /// assert_eq!(p2, point!(-0.70710677, 0.70710677, 0.0));
    /// assert_eq!(p1, p2);
    /// ```
    fn rotate(&mut self, radians: [Radians; 3]);
}

impl<T> Rotate for T
where
    T: TransformMatrix,
{
    fn rotate(&mut self, radians: [Radians; 3]) {
        let transform = self.get_transform_matrix_mut();
        *transform *= glam::Mat4::from_rotation_x(radians[0].0);
        *transform *= glam::Mat4::from_rotation_y(radians[1].0);
        *transform *= glam::Mat4::from_rotation_z(radians[2].0);
    }

    fn rotate_x(&mut self, radians: Radians) {
        let transform = self.get_transform_matrix_mut();
        *transform *= glam::Mat4::from_rotation_x(radians.0);
    }

    fn rotate_y(&mut self, radians: Radians) {
        let transform = self.get_transform_matrix_mut();
        *transform *= glam::Mat4::from_rotation_y(radians.0);
    }

    fn rotate_z(&mut self, radians: Radians) {
        let transform = self.get_transform_matrix_mut();
        *transform *= glam::Mat4::from_rotation_z(radians.0);
    }
}

impl Rotate for Point {
    fn rotate_x(&mut self, radians: Radians) {
        self.0 = glam::Mat4::from_rotation_x(radians.0) * self.0;
    }

    fn rotate_y(&mut self, radians: Radians) {
        self.0 = glam::Mat4::from_rotation_y(radians.0) * self.0;
    }

    fn rotate_z(&mut self, radians: Radians) {
        self.0 = glam::Mat4::from_rotation_z(radians.0) * self.0;
    }

    fn rotate(&mut self, radians: [Radians; 3]) {
        self.rotate_x(radians[0]);
        self.rotate_y(radians[1]);
        self.rotate_z(radians[2]);
    }
}

/// Scales an entity (makes it bigger or smaller).
pub trait Scale {
    /// Scales an entity (makes it bigger or smaller).
    fn scale(&mut self, scale: glam::Vec3);
}

/// Scales an entity (makes it bigger or smaller).
pub trait ScaleOrigin
where
    Self: HasOrigin + HasDirection,
{
    /// Scales an entity (makes it bigger or smaller).
    fn scale(&mut self, scale: glam::Vec3);
}

impl<T> ScaleOrigin for T
where
    T: HasOrigin + HasDirection,
{
    fn scale(&mut self, scale: glam::Vec3) {
        let scale = scale.extend(1.0);
        let origin = self.get_origin_mut();
        *origin = Point(origin.0 * scale);
        let direction = self.get_direction_mut();
        *direction = (direction.0 * scale).into();
    }
}

impl<T> Scale for T
where
    T: TransformMatrix,
{
    /// Scales an entity (makes it bigger or smaller).
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    /// let mut sphere = Sphere::test_sphere();
    /// scale!(sphere, 2, 3, 4);
    /// let transform = glam::mat4(
    ///    glam::vec4(2.0, 0.0, 0.0, 0.0),
    ///    glam::vec4(0.0, 3.0, 0.0, 0.0),
    ///    glam::vec4(0.0, 0.0, 4.0, 0.0),
    ///    glam::vec4(0.0, 0.0, 0.0, 1.0),
    /// );
    /// assert_eq!(sphere.transform, transform);
    /// assert_eq!(sphere.origin, point!());
    /// ```
    fn scale(&mut self, scale: glam::Vec3) {
        let transform = self.get_transform_matrix_mut();
        *transform *= glam::Mat4::from_scale(scale);
    }
}

// impl<T> Translate for T where T: Origin {
//     fn translate(&mut self, translation: glam::Vec3) {
//         let origin = self.get_origin_mut();
//         *origin = *origin + translation;
//     }
// }
// impl<T> Scale for T where T: Origin + Direction {
//     /// Scales an entity (makes it bigger or smaller).
//     /// ```rust
//     /// use engine::rt::*;
//     /// use glam;
//     /// let mut ray = Ray {
//     ///     origin: glam::vec4(1.0, 2.0, 3.0, 0.0),
//     ///     direction: glam::vec4(1.0, 0.0, 0.0, 0.0),
//     /// };
//     /// let scale = glam::vec4(2.0, 3.0, 4.0, 0.0);
//     /// ray.scale(scale);
//     /// assert_eq!(ray.origin, glam::vec4(2.0, 6.0, 12.0, 0.0));
//     /// assert_eq!(ray.direction, glam::vec4(2.0, 0.0, 0.0, 0.0));
//     /// ```
//     fn scale(&mut self, scale: glam::Vec4) {
//         let origin = self.get_origin_mut();
//         *origin = *origin * scale;
//         let direction = self.get_direction_mut();
//         *direction = *direction * scale;
//     }
// }

/// An intersection object as per the book.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Intersection<'a, T: ?Sized> {
    /// The object intersected.
    pub object: &'a T,
    /// The value of the intersection as per the book.
    pub value: f32,
}

/// Defines whether [`Self`] intersects with another object.
pub trait IntersectsWith<Object: ?Sized> {
    /// Returns distances to intersections from the origin of the ray.
    fn distance_to_intersections(&self, object: &Object) -> Vec<f32>;
}

/// Defines intersections as per the book.
pub trait IntersectionsWith<'a, TestObject: ?Sized, HitObject: ?Sized> {
    /// Returns intersection objects.
    fn intersections_with(&self, object: &'a TestObject) -> Vec<Intersection<'a, HitObject>>;
}

impl<'a, L, O> IntersectionsWith<'a, dyn WorldLike<L, O>, O> for Ray
where
    Ray: IntersectionsWith<'a, O, O>,
    O: std::fmt::Debug,
{
    /// Intersects a ray with a world.
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    /// let world = test_world();
    /// let ray = RayBuilder::default()
    ///     .origin(point!(0.0, 0.0, -5.0))
    ///     .direction(normalised_direction!(0.0, 0.0, 1.0))
    ///     .build().unwrap();
    //    /// let world_like = &world as &dyn WorldLike<dyn LightLike, dyn WorldObject>;
    /// let world_like = &world as &dyn WorldLike<_, _>;
    /// let intersections = ray.intersections_with(world_like);
    /// assert_eq!(intersections.len(), 4);
    /// assert_eq!(intersections[0].value, 4.0);
    /// assert_eq!(intersections[1].value, 4.5);
    /// assert_eq!(intersections[2].value, 5.5);
    /// assert_eq!(intersections[3].value, 6.0);
    /// ```
    fn intersections_with(&self, world: &'a dyn WorldLike<L, O>) -> Vec<Intersection<'a, O>> {
        let mut intersections: Vec<Intersection<'a, O>> = world
            .get_objects()
            .iter()
            .filter_map(|o| {
                let mut intersections = self.intersections_with(*o);
                intersections.retain(|a| a.value != std::f32::NAN && a.value >= 0.0);
                if intersections.is_empty() {
                    None
                } else {
                    Some(intersections)
                }
            })
            .flatten()
            .collect();
        intersections.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap());
        intersections
    }
}

impl<'a, T> IntersectionsWith<'a, T, T> for Ray
where
    Ray: IntersectsWith<T>,
{
    /// Ray successfully intersects two times, crossing the sphere.
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    /// let sphere = Sphere::test_sphere();
    /// let ray = RayBuilder::default()
    ///     .origin(point!(-5.0, 0.0, 0.0))
    ///     .direction(normalised_direction!(1.0, 0.0, 0.0))
    ///     .build().unwrap();
    /// let intersections = ray.intersections_with(&sphere);
    /// assert_eq!(intersections.len(), 2);
    /// assert_eq!(intersections[0].object, &sphere);
    /// assert!(intersections[0].value >= 4.0 - std::f32::EPSILON);
    /// assert!(intersections[0].value <= 4.0 + std::f32::EPSILON);
    /// assert_eq!(intersections[1].object, &sphere);
    /// assert!(intersections[1].value >= 6.0 - std::f32::EPSILON);
    /// assert!(intersections[1].value <= 6.0 + std::f32::EPSILON);
    /// ```
    fn intersections_with(&self, object: &'a T) -> Vec<Intersection<'a, T>> {
        let mut distances: Vec<Intersection<'a, T>> = self
            .distance_to_intersections(object)
            .into_iter()
            .map(|value| Intersection { object, value })
            .collect();
        distances.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap());
        distances
    }
}

/// Returns only hits, not all intersections.
pub trait Hits<'a, Object> {
    /// Returns all the hits.
    fn hits(&self, object: &'a Object) -> Vec<Intersection<'a, Object>>;
}

impl<'a, T, Object> Hits<'a, Object> for T
where
    T: IntersectionsWith<'a, Object, Object>,
{
    fn hits(&self, object: &'a Object) -> Vec<Intersection<'a, Object>> {
        self.intersections_with(object)
            .into_iter()
            .filter(|v| v.value >= 0.0)
            .collect()
    }
}

/// Implementors of the trait return a color with which an object is colored with after having reflected
/// other lights and objects.
pub trait ColoredHit<'b, O: ?Sized> {
    /// Returns a color with which a ray reflects back to the eye from various lights and objects.
    /// In other words, what an eye should see when looking at the object and a series of light rays
    /// reflect from the object and hit the eye.
    fn colored_hit(&self, object: &'b O) -> ColorRGBA;
}

/// A ray-hittable or drawable object.
// pub trait Hittable<'a>: Origin + TransformMatrix where Ray: Hits<'a, Self> {}
pub trait Hittable {}
impl<'a, T> Hittable for T where Ray: Hits<'a, T> + IntersectsWith<T> + IntersectionsWith<'a, T, T> {}

/// Indicates that an object may be a part of the world.
pub trait WorldObject:
    HasOrigin + TransformMatrix + Hittable + SphereLike + std::fmt::Debug
{
}
impl<T> WorldObject for T where
    T: HasOrigin + TransformMatrix + Hittable + SphereLike + std::fmt::Debug
{
}

impl<L, O> IntersectsWith<dyn WorldLike<L, O>> for Ray
where
    L: LightLike,
    O: WorldObject,
{
    /// Intersects a ray with a world.
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    /// let world = test_world();
    /// let ray = RayBuilder::default()
    ///     .origin(point!(0.0, 0.0, -5.0))
    ///     .direction(normalised_direction!(0.0, 0.0, 1.0))
    ///     .build().unwrap();
    //    /// let world_like = &world as &dyn WorldLike<dyn LightLike, dyn WorldObject>;
    /// let world_like = &world as &dyn WorldLike<_, _>;
    /// let distances_to_intersections = ray.distance_to_intersections(world_like);
    /// assert_eq!(distances_to_intersections.len(), 4);
    /// assert_eq!(distances_to_intersections[0], 4.0);
    /// assert_eq!(distances_to_intersections[1], 4.5);
    /// assert_eq!(distances_to_intersections[2], 5.5);
    /// assert_eq!(distances_to_intersections[3], 6.0);
    /// ```
    fn distance_to_intersections(&self, world: &dyn WorldLike<L, O>) -> Vec<f32> {
        let mut distances: Vec<f32> = world
            .get_objects()
            .iter()
            .filter_map(|o| {
                let mut intersections = self.distance_to_intersections(*o);
                intersections.retain(|a| *a != std::f32::NAN && *a >= 0.0);
                if intersections.is_empty() {
                    None
                } else {
                    Some(intersections)
                }
            })
            .flatten()
            .collect();
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        distances
    }
}

impl<'a, 'b, L, O> ColoredHit<'a, dyn WorldLike<L, O> + 'b> for Ray
where
    L: LightLike,
    O: WorldObject,
    Ray: IntersectionsWith<'a, dyn WorldLike<L, O> + 'b, O>,
{
    /// The color when a ray misses.
    ///
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    /// let world = test_world();
    /// let ray = RayBuilder::default()
    ///     .origin(point!(0.0, 0.0, -5.0))
    ///     .direction(normalised_direction!(0.0, 1.0, 0.0))
    ///     .build().unwrap();
    /// let world_like = &world as &dyn WorldLike<_, _>;
    /// assert_eq!(ray.colored_hit(world_like), ColorRGBA::black());
    /// ```
    ///
    /// The color when a ray hits.
    ///
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    /// let world = test_world();
    /// let ray = RayBuilder::default()
    ///     .origin(point!(0.0, 0.0, -5.0))
    ///     .direction(normalised_direction!(0.0, 0.0, 1.0))
    ///     .build().unwrap();
    /// let world_like = &world as &dyn WorldLike<_, _>;
    /// assert_eq!(ray.colored_hit(world_like), glam::vec4(0.3806612, 0.47582647, 0.2854959, 1.0).into());
    /// ```
    ///
    /// The color with an intersection behind the ray. Page 181.
    ///
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    /// let mut world = test_world();
    /// world.objects[0].get_material_mut().ambient = 1.0;
    /// world.objects[1].get_material_mut().ambient = 1.0;
    /// let inner_object_color = world.objects[1].get_material().color;
    /// let ray = RayBuilder::default()
    ///     .origin(point!(0.0, 0.0, 0.75))
    ///     .direction(normalised_direction!(0.0, 0.0, -1.0))
    ///     .build().unwrap();
    /// let world_like = &world as &dyn WorldLike<_, _>;
    /// assert_eq!(ray.colored_hit(world_like), inner_object_color);
    /// ```
    fn colored_hit(&self, world: &'a (dyn WorldLike<L, O> + 'b)) -> ColorRGBA {
        self.intersections_with(world)
            .into_iter()
            .next()
            .map(|i| PreparedComputations::new(i, *self))
            .map(|pc| world.shade_hit(&pc))
            .unwrap_or_else(|| ColorRGBA::black())
    }
}

impl<T> IntersectsWith<T> for Ray
where
    T: SphereLike,
{
    /// Ray successfully intersects two times, crossing the sphere.
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    /// let sphere = Sphere::test_sphere();
    /// let ray = RayBuilder::default()
    ///     .origin(point!(-5.0, 0.0, 0.0))
    ///     .direction(normalised_direction!(1.0, 0.0, 0.0))
    ///     .build().unwrap();
    /// assert_eq!(ray.distance_to_intersections(&sphere), vec![4.0, 6.0]);
    /// ```
    ///
    /// Ray only touches the sphere one time.
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    /// let sphere = Sphere::test_sphere();
    /// let ray = RayBuilder::default()
    ///     .origin(point!(-5.0, 1.0, 0.0))
    ///     .direction(normalised_direction!(1.0, 0.0, 0.0))
    ///     .build().unwrap();
    /// assert_eq!(ray.distance_to_intersections(&sphere), vec![5.0, 5.0]);
    /// ```
    ///
    /// Ray completely misses the sphere.
    /// ```rust
    /// use engine::*;
    /// use engine::rt::*;
    /// use glam;
    /// let sphere = Sphere::test_sphere();
    /// let ray = RayBuilder::default()
    ///     .origin(point!(-5.0, 2.0, 0.0))
    ///     .direction(normalised_direction!(1.0, 0.0, 0.0))
    ///     .build().unwrap();
    /// assert_eq!(ray.distance_to_intersections(&sphere), vec![]);
    /// ```
    fn distance_to_intersections(&self, sphere: &T) -> Vec<f32> {
        let mut ray = *self;
        let sphere_transform = *sphere.get_transform_matrix();
        let sphere_origin = *sphere.get_origin();
        let sphere_radius = sphere.get_radius();
        let (sphere_scale, _, sphere_translation) =
            sphere_transform.inverse().to_scale_rotation_translation();
        ray.scale(sphere_scale);
        ray.translate_origin(sphere_translation);
        let sphere_to_ray = ray.origin.0 - sphere_origin.0;
        let a = ray.direction.0.length_squared();
        let half_b = ray.direction.0.dot(sphere_to_ray);
        let c = sphere_to_ray.length_squared() - sphere_radius * sphere_radius;
        let d = half_b * half_b - a * c;
        if d < 0.0 {
            Vec::new()
        } else {
            let t1 = (-half_b - d.sqrt()) / a;
            let t2 = (-half_b + d.sqrt()) / a;
            vec![t1, t2]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use glam;

    #[test]
    fn test_chaining_transformations() {
        // This shows that the glam works correctly, according to what is expected.
        // Page 107.
        let teapot = point!(1, 0, 1).0;
        let a = glam::Mat4::from_rotation_x(degree!(90).0);
        let b = glam::Mat4::from_scale(glam::vec3(5.0, 5.0, 5.0));
        let c = glam::Mat4::from_translation(glam::vec3(10.0, 5.0, 7.0));
        assert_eq!(c * (b * (a * teapot)), (c * b * a) * teapot);
        let teapot = (c * b * a) * teapot;
        assert_eq!(teapot, point!(15, 0, 7).0);

        // Here we want to reproduce the same using the interface.
        let mut teapot2 = point!(1, 0, 1);
        teapot2.rotate_x(degree!(90));
        teapot2.scale(glam::vec3(5.0, 5.0, 5.0));
        teapot2.translate(glam::vec3(10.0, 5.0, 7.0));
        assert_eq!(teapot2.0, teapot);
    }
}
