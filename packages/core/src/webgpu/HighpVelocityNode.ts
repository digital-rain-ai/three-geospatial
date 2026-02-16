import {
  DynamicDrawUsage,
  InstancedBufferAttribute,
  Matrix4,
  type Object3D
} from 'three'
import {
  attribute,
  instancedBufferAttribute,
  mat4,
  nodeImmutable,
  positionPrevious,
  positionView,
  sub,
  uniform,
  vec4
} from 'three/tsl'
import {
  NodeUpdateType,
  TempNode,
  type NodeBuilder,
  type NodeFrame
} from 'three/webgpu'

const PREV_ATTR0 = 'instanceMatrixPrevious0'
const PREV_ATTR1 = 'instanceMatrixPrevious1'
const PREV_ATTR2 = 'instanceMatrixPrevious2'
const PREV_ATTR3 = 'instanceMatrixPrevious3'

interface PrevCols {
  a0: InstancedBufferAttribute
  a1: InstancedBufferAttribute
  a2: InstancedBufferAttribute
  a3: InstancedBufferAttribute
}

interface InstancedObject extends Object3D {
  geometry: {
    setAttribute: (name: string, attribute: InstancedBufferAttribute) => void
  }
  instanceMatrix?: InstancedBufferAttribute
  material?: {
    positionNode?: {
      attribute?: InstancedBufferAttribute
    }
  }
  count?: number
  instanceCount?: number
}

type InstancedMode = 'matrix' | 'position' | null

export class HighpVelocityNode extends TempNode {
  static override get type(): string {
    return 'HighpVelocityNode'
  }

  projectionMatrix?: Matrix4 | null

  private readonly currentProjectionMatrix = uniform('mat4')
  private readonly previousProjectionMatrix = uniform('mat4')

  private readonly currentModelViewMatrix = uniform('mat4')
  private readonly previousModelViewMatrix = uniform('mat4')
  private readonly objectModelViewMatrices = new WeakMap<Object3D, Matrix4>()

  // Per-instanced-object "previous instance matrix" as 4 vec4 columns
  private readonly prevCols = new WeakMap<InstancedObject, PrevCols>()
  private readonly seededPrev = new WeakSet<InstancedObject>()
  private readonly prevPositions = new WeakMap<InstancedObject, InstancedBufferAttribute>()
  private readonly seededPrevPositions = new WeakSet<InstancedObject>()

  constructor() {
    super('vec3')

    // Sequence:
    // - updateBefore() for the first object
    // - update() for the current frame
    // - updateAfter() for the first object
    // - updateBefore() for the next object
    // - updateAfter() for the next object
    // - ...
    this.updateType = NodeUpdateType.FRAME
    this.updateBeforeType = NodeUpdateType.OBJECT
    this.updateAfterType = NodeUpdateType.OBJECT
  }

  setProjectionMatrix(value: Matrix4 | null): this {
    this.projectionMatrix = value
    return this
  }

  // Executed once per frame:
  override update({ camera }: NodeFrame): void {
    if (camera == null) {
      return
    }
    const {
      currentProjectionMatrix: current,
      previousProjectionMatrix: previous
    } = this

    const projectionMatrix = this.projectionMatrix ?? camera.projectionMatrix
    if (previous.value == null) {
      previous.value = new Matrix4().copy(projectionMatrix)
    } else {
      previous.value.copy(current.value)
    }
    current.value.copy(projectionMatrix)
  }

  // --- helpers ---

  private static getPositionSourceAttribute(object: Object3D): InstancedBufferAttribute | undefined {
    const fromUserData = (object as any)?.material?.userData?.highpVelocityPositionAttribute
    if (fromUserData?.isInstancedBufferAttribute === true && fromUserData.itemSize >= 3) {
      return fromUserData as InstancedBufferAttribute
    }

    const candidate = (object as any)?.material?.positionNode?.attribute
    if (candidate?.isInstancedBufferAttribute === true && candidate.itemSize >= 3) {
      return candidate as InstancedBufferAttribute
    }

    return undefined
  }

  private static getInstancedMode(object: Object3D): InstancedMode {
    if ((object as any)?.isInstancedMesh === true) {
      return 'matrix'
    }

    if (
      (object as any)?.isSprite === true &&
      HighpVelocityNode.getPositionSourceAttribute(object) != null &&
      typeof (object as any)?.geometry?.setAttribute === 'function'
    ) {
      return 'position'
    }

    return null
  }

  private static isSupportedInstancedObject(object: Object3D): object is InstancedObject {
    return HighpVelocityNode.getInstancedMode(object) != null
  }

  private static getInstanceCount(object: InstancedObject): number {
    // Robust across Three versions / wrappers
    const positionSource = HighpVelocityNode.getPositionSourceAttribute(object)
    const fromMesh =
      (positionSource as any)?.count ??
      (object as any).count ??
      (object as any).instanceCount ??
      (object.instanceMatrix as any)?.count

    return typeof fromMesh === 'number' ? fromMesh : 0
  }

  private static getPositionRequiredCount(
    object: InstancedObject,
    src: InstancedBufferAttribute
  ): number {
    const objectCount =
      typeof (object as any).count === 'number'
        ? (object as any).count as number
        : typeof (object as any).instanceCount === 'number'
          ? (object as any).instanceCount as number
          : 0

    return Math.max(src.count, objectCount)
  }

  private ensurePrevPositionsAllocated(object: InstancedObject): void {
    const src = HighpVelocityNode.getPositionSourceAttribute(object)
    if (src == null) {
      return
    }

    const count = HighpVelocityNode.getPositionRequiredCount(object, src)
    let prev = this.prevPositions.get(object)
    const needsRealloc = prev == null || prev.count < count

    if (needsRealloc) {
      prev = new InstancedBufferAttribute(new Float32Array(count * 3), 3)
      prev.usage = DynamicDrawUsage
      this.prevPositions.set(object, prev)
      this.seededPrevPositions.delete(object)
    }

    if (!this.seededPrevPositions.has(object)) {
      prev = this.prevPositions.get(object)
      if (prev != null) {
        const source = src.array as Float32Array
        const target = prev.array as Float32Array
        const copyCount = Math.min(src.count, prev.count)
        for (let i = 0; i < copyCount; i++) {
          const s = i * src.itemSize
          const d = i * 3
          target[d + 0] = source[s + 0]
          target[d + 1] = source[s + 1]
          target[d + 2] = source[s + 2]
        }
        prev.needsUpdate = true
      }
      this.seededPrevPositions.add(object)
    }
  }

  private ensurePrevColsAllocated(object: InstancedObject): void {
    const geom = object.geometry
    const count = HighpVelocityNode.getInstanceCount(object)

    let cols = this.prevCols.get(object)
    const needsRealloc =
      cols?.a0.count !== count ||
      cols?.a1.count !== count ||
      cols?.a2.count !== count ||
      cols?.a3.count !== count

    if (needsRealloc) {
      const makeCol = (name: string): InstancedBufferAttribute => {
        const attr = new InstancedBufferAttribute(new Float32Array(count * 4), 4)
        attr.usage = DynamicDrawUsage
        geom.setAttribute(name, attr)
        return attr
      }

      cols = {
        a0: makeCol(PREV_ATTR0),
        a1: makeCol(PREV_ATTR1),
        a2: makeCol(PREV_ATTR2),
        a3: makeCol(PREV_ATTR3)
      }

      this.prevCols.set(object, cols)
      this.seededPrev.delete(object) // force reseed after realloc
    }

    // Seed once so first frame has 0 velocity
    if (!this.seededPrev.has(object)) {
      cols = this.prevCols.get(object)
      const src = (object as any).instanceMatrix as InstancedBufferAttribute | undefined
      if (src != null && cols != null) {
        const S = src.array as Float32Array
        const a0 = cols.a0.array as Float32Array
        const a1 = cols.a1.array as Float32Array
        const a2 = cols.a2.array as Float32Array
        const a3 = cols.a3.array as Float32Array

        for (let i = 0; i < count; i++) {
          const s = i * 16
          const d = i * 4
          // Copy columns (matches how Three uses instanceMatrix in the vertex chunk)
          a0[d + 0] = S[s + 0];  a0[d + 1] = S[s + 1];  a0[d + 2] = S[s + 2];  a0[d + 3] = S[s + 3]
          a1[d + 0] = S[s + 4];  a1[d + 1] = S[s + 5];  a1[d + 2] = S[s + 6];  a1[d + 3] = S[s + 7]
          a2[d + 0] = S[s + 8];  a2[d + 1] = S[s + 9];  a2[d + 2] = S[s +10];  a2[d + 3] = S[s +11]
          a3[d + 0] = S[s +12];  a3[d + 1] = S[s +13];  a3[d + 2] = S[s +14];  a3[d + 3] = S[s +15]
        }

        cols.a0.needsUpdate = cols.a1.needsUpdate = cols.a2.needsUpdate = cols.a3.needsUpdate = true
      }
      this.seededPrev.add(object)
    }
  }

  // Executed once per object before rendering:
  override updateBefore({ object, camera }: NodeFrame): void {
    if (object == null || camera == null) {
      return
    }
    const {
      currentModelViewMatrix: current,
      previousModelViewMatrix: previous,
      objectModelViewMatrices: matrices
    } = this

    current.value.multiplyMatrices(
      camera.matrixWorldInverse,
      object.matrixWorld
    )
    previous.value = matrices.get(object) ?? current.value

    // For supported instanced objects, make sure previous columns exist *now* (also guards first compile)
    const mode = HighpVelocityNode.getInstancedMode(object)
    if (mode === 'matrix') {
      const instancedObject = object as InstancedObject
      this.ensurePrevColsAllocated(instancedObject)
    } else if (mode === 'position') {
      const instancedObject = object as InstancedObject
      this.ensurePrevPositionsAllocated(instancedObject)
    }
  }

  // Executed once per object after rendering:
  override updateAfter({ object }: NodeFrame): void {
    if (object == null) {
      return
    }
    const {
      currentModelViewMatrix: current,
      objectModelViewMatrices: matrices
    } = this

    let matrix = matrices.get(object)
    if (matrix == null) {
      matrix = new Matrix4()
      matrices.set(object, matrix)
    }
    matrix.copy(current.value)

    // For instanced, update previous columns from current instanceMatrix
    const mode = HighpVelocityNode.getInstancedMode(object)
    if (mode === 'matrix') {
      const instancedObject = object as InstancedObject
      this.ensurePrevColsAllocated(instancedObject)

      const cols = this.prevCols.get(instancedObject)
      const src = (instancedObject as any).instanceMatrix as InstancedBufferAttribute | undefined
      if (cols != null && src != null) {
        const count = HighpVelocityNode.getInstanceCount(instancedObject)
        const S = src.array as Float32Array
        const a0 = cols.a0.array as Float32Array
        const a1 = cols.a1.array as Float32Array
        const a2 = cols.a2.array as Float32Array
        const a3 = cols.a3.array as Float32Array

        for (let i = 0; i < count; i++) {
          const s = i * 16
          const d = i * 4
          a0[d + 0] = S[s + 0];  a0[d + 1] = S[s + 1];  a0[d + 2] = S[s + 2];  a0[d + 3] = S[s + 3]
          a1[d + 0] = S[s + 4];  a1[d + 1] = S[s + 5];  a1[d + 2] = S[s + 6];  a1[d + 3] = S[s + 7]
          a2[d + 0] = S[s + 8];  a2[d + 1] = S[s + 9];  a2[d + 2] = S[s +10];  a2[d + 3] = S[s +11]
          a3[d + 0] = S[s +12];  a3[d + 1] = S[s +13];  a3[d + 2] = S[s +14];  a3[d + 3] = S[s +15]
        }

        cols.a0.needsUpdate = cols.a1.needsUpdate = cols.a2.needsUpdate = cols.a3.needsUpdate = true
      }
    } else if (mode === 'position') {
      const instancedObject = object as InstancedObject
      this.ensurePrevPositionsAllocated(instancedObject)

      const prev = this.prevPositions.get(instancedObject)
      const src = HighpVelocityNode.getPositionSourceAttribute(instancedObject)
      if (prev != null && src != null) {
        const count = Math.min(src.count, prev.count)
        const source = src.array as Float32Array
        const target = prev.array as Float32Array

        for (let i = 0; i < count; i++) {
          const s = i * src.itemSize
          const d = i * 3
          target[d + 0] = source[s + 0]
          target[d + 1] = source[s + 1]
          target[d + 2] = source[s + 2]
        }

        prev.needsUpdate = true
      }
    }
  }

  override setup(builder: NodeBuilder) : unknown {
    const obj = (builder as any)?.object as Object3D | undefined

    const instancedMode = obj == null ? null : HighpVelocityNode.getInstancedMode(obj)

    // Ensure prev attributes exist before the shader is compiled (first build)
    if (instancedMode === 'matrix') {
      this.ensurePrevColsAllocated(obj as InstancedObject)
    } else if (instancedMode === 'position') {
      this.ensurePrevPositionsAllocated(obj as InstancedObject)
    }

    // --- Current clip position ---
    // Use positionView so instancing is already applied by TSL's transform chain.
    const currentClip = this.currentProjectionMatrix
      .mul(vec4(positionView, 1))
      .toVertexStage()

    // --- Previous clip position ---
    // For non-instanced objects, the classic MV * positionPrevious is enough.
    // For supported instanced objects, multiply by our per-instance previous matrix (four vec4 columns).
    let previousPath = this.previousProjectionMatrix.mul(this.previousModelViewMatrix)

    if (instancedMode === 'matrix') {
      const c0 = attribute(PREV_ATTR0, 'vec4')
      const c1 = attribute(PREV_ATTR1, 'vec4')
      const c2 = attribute(PREV_ATTR2, 'vec4')
      const c3 = attribute(PREV_ATTR3, 'vec4')
      previousPath = previousPath.mul(mat4(c0.x as any, c0.y as any, c0.z as any, c0.w as any, c1.x as any, c1.y as any, c1.z as any, c1.w as any, c2.x as any, c2.y as any, c2.z as any, c2.w as any, c3.x as any, c3.y as any, c3.z as any, c3.w as any))
    } else if (instancedMode === 'position') {
      const prevPosition = this.prevPositions.get(obj as InstancedObject)
      if (prevPosition == null) {
        const previousClip = previousPath.mul(positionPrevious).toVertexStage()
        const currentNDC = currentClip.xyz.div(currentClip.w)
        const previousNDC = previousClip.xyz.div(previousClip.w)
        return sub(currentNDC, previousNDC)
      }

      previousPath = previousPath.mul(vec4(instancedBufferAttribute(prevPosition, 'vec3'), 1))
      const previousClip = previousPath.toVertexStage()
      const currentNDC = currentClip.xyz.div(currentClip.w)
      const previousNDC = previousClip.xyz.div(previousClip.w)
      return sub(currentNDC, previousNDC)
    }

    const previousClip = previousPath.mul(positionPrevious).toVertexStage()

    // Perspective divisions cannot be performed in the vertex shader.
    // See: http://john-chapman-graphics.blogspot.com/2013/01/per-object-motion-blur.html
    const currentNDC = currentClip.xyz.div(currentClip.w)
    const previousNDC = previousClip.xyz.div(previousClip.w)

    return sub(currentNDC, previousNDC)
  }
}

export const highpVelocity = /*#__PURE__*/ nodeImmutable(HighpVelocityNode)