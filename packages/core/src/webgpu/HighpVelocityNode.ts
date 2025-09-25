import {
  DynamicDrawUsage,
  InstancedBufferAttribute,
  type InstancedMesh,
  Matrix4,
  type Object3D
} from 'three'
import {
  attribute,
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

type PrevCols = {
  a0: InstancedBufferAttribute
  a1: InstancedBufferAttribute
  a2: InstancedBufferAttribute
  a3: InstancedBufferAttribute
}

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

  // Per-instanced-mesh "previous instance matrix" as 4 vec4 columns
  private readonly prevCols = new WeakMap<InstancedMesh, PrevCols>()
  private readonly seededPrev = new WeakSet<InstancedMesh>()

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

  private getInstanceCount(mesh: InstancedMesh): number {
    // Robust across Three versions / wrappers
    const fromMesh =
      (mesh as any).count ??
      (mesh as any).instanceCount ??
      (mesh.instanceMatrix as any)?.count

    return typeof fromMesh === 'number' ? fromMesh : 0
  }

  private ensurePrevColsAllocated(mesh: InstancedMesh): void {
    const geom = mesh.geometry
    const count = this.getInstanceCount(mesh)

    let cols = this.prevCols.get(mesh)
    const needsRealloc =
      !cols ||
      cols.a0.count !== count ||
      cols.a1.count !== count ||
      cols.a2.count !== count ||
      cols.a3.count !== count

    if (needsRealloc) {
      const makeCol = (name: string) => {
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

      this.prevCols.set(mesh, cols)
      this.seededPrev.delete(mesh) // force reseed after realloc
    }

    // Seed once so first frame has 0 velocity
    if (!this.seededPrev.has(mesh)) {
      const src = (mesh as any).instanceMatrix as InstancedBufferAttribute | undefined
      if (src && cols) {
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
      this.seededPrev.add(mesh)
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

    // For InstancedMesh, make sure previous columns exist *now* (also guards first compile)
    if ((object as any).isInstancedMesh === true) {
      this.ensurePrevColsAllocated(object as InstancedMesh)
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
    if ((object as any).isInstancedMesh === true) {
      const mesh = object as InstancedMesh
      this.ensurePrevColsAllocated(mesh)

      const cols = this.prevCols.get(mesh)!
      const src = (mesh as any).instanceMatrix as InstancedBufferAttribute | undefined
      if (src) {
        const count = this.getInstanceCount(mesh)
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
    }
  }

  override setup(builder: NodeBuilder) : unknown {
    const obj = (builder as any)?.object as Object3D | undefined
    const isInstanced = (obj as any)?.isInstancedMesh === true

    // Ensure prev attributes exist before the shader is compiled (first build)
    if (isInstanced) this.ensurePrevColsAllocated(obj as InstancedMesh)

    // --- Current clip position ---
    // Use positionView so instancing is already applied by TSL's transform chain.
    const currentClip = this.currentProjectionMatrix
      .mul(vec4(positionView, 1))
      .toVertexStage()

    // --- Previous clip position ---
    // For non-instanced objects, the classic MV * positionPrevious is enough.
    // For InstancedMesh, multiply by our per-instance previous matrix (four vec4 columns).
    let previousPath = this.previousProjectionMatrix.mul(this.previousModelViewMatrix)

    if (isInstanced) {
      const c0 = attribute(PREV_ATTR0, 'vec4')
      const c1 = attribute(PREV_ATTR1, 'vec4')
      const c2 = attribute(PREV_ATTR2, 'vec4')
      const c3 = attribute(PREV_ATTR3, 'vec4')
      previousPath = previousPath.mul(mat4(c0.x as any, c0.y as any, c0.z as any, c0.w as any, c1.x as any, c1.y as any, c1.z as any, c1.w as any, c2.x as any, c2.y as any, c2.z as any, c2.w as any, c3.x as any, c3.y as any, c3.z as any, c3.w as any))
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