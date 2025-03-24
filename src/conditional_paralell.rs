#[cfg(feature = "parallel")]
pub use rayon::prelude;

#[cfg(not(feature = "parallel"))]
pub mod prelude {
    pub use std::iter::Iterator as ParallelIterator;
    pub trait ParallelBridge: Sized {
        fn par_bridge(self) -> Self;
    }

    impl<T: Iterator + Send> ParallelBridge for T {
        fn par_bridge(self) -> Self {
            self
        }
    }

    pub trait IntoParallelIterator: Sized {
        type Item;
        type Iter: Iterator<Item = Self::Item>;

        fn into_par_iter(self) -> Self::Iter;
    }

    impl<I> IntoParallelIterator for I
    where
        I: IntoIterator,
    {
        type Item = I::Item;
        type Iter = I::IntoIter;

        fn into_par_iter(self) -> Self::Iter {
            self.into_iter()
        }
    }

    pub trait IntoParallelRefMutIterator<'data> {
        type Iter: ParallelIterator<Item = Self::Item>;

        /// The type of item that will be produced; this is typically an
        /// `&'data mut T` reference.
        type Item: 'data;

        fn par_iter_mut(&'data mut self) -> Self::Iter;
    }

    impl<'data, I: 'data + ?Sized> IntoParallelRefMutIterator<'data> for I
    where
        &'data mut I: IntoParallelIterator,
    {
        type Iter = <&'data mut I as IntoParallelIterator>::Iter;
        type Item = <&'data mut I as IntoParallelIterator>::Item;

        fn par_iter_mut(&'data mut self) -> Self::Iter {
            self.into_par_iter()
        }
    }

    pub trait IntoParallelRefIterator<'data> {
        type Item: 'data;
        type Iter: ParallelIterator<Item = Self::Item>;

        fn par_iter(&'data self) -> Self::Iter;
    }

    impl<'data, I: 'data + ?Sized> IntoParallelRefIterator<'data> for I
    where
        &'data I: IntoParallelIterator,
    {
        type Iter = <&'data I as IntoParallelIterator>::Iter;
        type Item = <&'data I as IntoParallelIterator>::Item;

        fn par_iter(&'data self) -> Self::Iter {
            self.into_par_iter()
        }
    }
}
