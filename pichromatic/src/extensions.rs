use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt::Debug;

/// Trait for types that can be stored inside [`Extensions`].
///
/// Automatically implemented for all types that implement `Clone + Send + Sync + Debug + PartialEq + 'static`.
pub trait ExtensionValue: Any + Send + Sync + Debug {
    fn clone_box(&self) -> Box<dyn ExtensionValue>;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
    fn is_equal(&self, other: &dyn ExtensionValue) -> bool;
}

impl<T: Clone + Send + Sync + Debug + PartialEq + 'static> ExtensionValue for T {
    fn clone_box(&self) -> Box<dyn ExtensionValue> {
        Box::new(self.clone())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
    fn is_equal(&self, other: &dyn ExtensionValue) -> bool {
        if let Some(other_t) = other.as_any().downcast_ref::<T>() {
            self == other_t
        } else {
            false
        }
    }
}

/// A type-map container for dynamically publishing and reading typed metadata/context
/// across pipeline stages.
#[derive(Default)]
pub struct Extensions {
    map: HashMap<TypeId, Box<dyn ExtensionValue>>,
}

impl Clone for Extensions {
    fn clone(&self) -> Self {
        let mut map = HashMap::with_capacity(self.map.len());
        for (&type_id, val) in &self.map {
            map.insert(type_id, val.clone_box());
        }
        Self { map }
    }
}

impl Debug for Extensions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Extensions")
            .field("entries", &self.map.len())
            .finish()
    }
}

impl PartialEq for Extensions {
    fn eq(&self, other: &Self) -> bool {
        if self.map.len() != other.map.len() {
            return false;
        }
        for (type_id, val) in &self.map {
            match other.map.get(type_id) {
                Some(other_val) => {
                    if !val.is_equal(other_val.as_ref()) {
                        return false;
                    }
                }
                None => return false,
            }
        }
        true
    }
}

impl Extensions {
    /// Create a new, empty `Extensions` container.
    #[inline]
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Insert a typed value into the container. Returns the previous value if present.
    pub fn insert<T: Clone + Send + Sync + Debug + PartialEq + 'static>(
        &mut self,
        val: T,
    ) -> Option<T> {
        self.map
            .insert(TypeId::of::<T>(), Box::new(val))
            .and_then(|boxed| boxed.into_any().downcast::<T>().ok().map(|b| *b))
    }

    /// Get a shared reference to a typed value stored in the container.
    pub fn get<T: 'static>(&self) -> Option<&T> {
        self.map
            .get(&TypeId::of::<T>())
            .and_then(|boxed| boxed.as_any().downcast_ref::<T>())
    }

    /// Get a mutable reference to a typed value stored in the container.
    pub fn get_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.map
            .get_mut(&TypeId::of::<T>())
            .and_then(|boxed| boxed.as_any_mut().downcast_mut::<T>())
    }

    /// Remove a typed value from the container.
    pub fn remove<T: 'static>(&mut self) -> Option<T> {
        self.map
            .remove(&TypeId::of::<T>())
            .and_then(|boxed| boxed.into_any().downcast::<T>().ok().map(|b| *b))
    }

    /// Check whether a type is present in the container.
    pub fn contains<T: 'static>(&self) -> bool {
        self.map.contains_key(&TypeId::of::<T>())
    }

    /// Clear all extensions.
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Number of items stored.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct CustomTag(String);

    #[derive(Clone, Debug, PartialEq)]
    struct Gain(f32);

    #[test]
    fn test_extensions_basic() {
        let mut ext = Extensions::new();
        assert!(ext.is_empty());
        assert_eq!(ext.len(), 0);

        ext.insert(Gain(2.5));
        assert_eq!(ext.len(), 1);
        assert!(!ext.is_empty());
        assert!(ext.contains::<Gain>());
        assert!(!ext.contains::<CustomTag>());

        assert_eq!(ext.get::<Gain>(), Some(&Gain(2.5)));

        if let Some(g) = ext.get_mut::<Gain>() {
            g.0 = 4.0;
        }
        assert_eq!(ext.get::<Gain>(), Some(&Gain(4.0)));

        ext.insert(CustomTag("test".to_string()));
        assert_eq!(ext.len(), 2);
        assert_eq!(ext.get::<CustomTag>(), Some(&CustomTag("test".to_string())));

        let cloned = ext.clone();
        assert_eq!(ext, cloned);

        assert_eq!(ext.remove::<Gain>(), Some(Gain(4.0)));
        assert_eq!(ext.len(), 1);
        assert!(!ext.contains::<Gain>());
        assert_ne!(ext, cloned);

        ext.clear();
        assert!(ext.is_empty());
    }
}
